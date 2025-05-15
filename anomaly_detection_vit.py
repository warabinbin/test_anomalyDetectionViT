import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Yu Gothic'

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 画像前処理の強化（データ拡張を含む）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 大きめにリサイズしてからクロップ
    transforms.RandomCrop(224),     # ランダムクロップ
    transforms.RandomHorizontalFlip(),  # 水平反転
    transforms.RandomRotation(10),   # 回転
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 色調整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNetの正規化
])

# テスト用の変換（データ拡張なし）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder, transform=test_transform, augment=False, num_augmentations=4):
    """フォルダから画像を読み込む関数（データ拡張オプション付き）"""
    images = []
    filenames = []
    
    if not os.path.exists(folder):
        print(f"Error: フォルダ '{folder}' が存在しません。")
        return torch.tensor([]), []
    
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not files:
        print(f"Warning: フォルダ '{folder}' に画像ファイルが見つかりません。")
        return torch.tensor([]), []
    
    for filename in tqdm(files, desc="画像読み込み"):
        try:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('RGB')
            
            # 基本の変換
            transformed_img = transform(img)
            images.append(transformed_img)
            filenames.append(filename)
            
            # 訓練データに対してはデータ拡張を実施
            if augment:
                for i in range(num_augmentations):
                    augmented_img = train_transform(img)  # 毎回異なる拡張が適用される
                    images.append(augmented_img)
                    filenames.append(f"{filename}_aug{i}")
        except Exception as e:
            print(f"Warning: 画像 '{filename}' の読み込みに失敗しました: {str(e)}")
    
    if not images:
        return torch.tensor([]), []
    
    return torch.stack(images), filenames

# 改良版ViT特徴抽出器（中間層からも特徴を取得）
class ImprovedViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Identity()  # 出力を特徴ベクトルに
        
        # 中間層にアクセスするためのフック
        self.hook_features = []
        self.hooks = []
        
        # 最後から3つのTransformerブロックにフックを追加
        for block in self.model.blocks[-3:]:
            self.hooks.append(block.register_forward_hook(self._get_features))
    
    def _get_features(self, module, input, output):
        # CLSトークンの特徴量を保存
        self.hook_features.append(output[:, 0])
    
    def forward(self, x):
        self.hook_features = []
        feat = self.model(x)
        
        # 中間層のCLSトークンを連結
        if self.hook_features:
            all_features = torch.cat([feat] + self.hook_features, dim=1)
            return all_features
        
        return feat
    
    def __del__(self):
        # フックを解除
        for hook in self.hooks:
            hook.remove()

def compute_adaptive_threshold(normal_scores, method='percentile', **kwargs):
    """
    適応的な閾値計算
    
    Args:
        normal_scores: 正常データのスコア配列
        method: 閾値決定方法 ('percentile', 'zscore', 'iqr', 'gmm')
        **kwargs: 各メソッドのパラメータ
    
    Returns:
        閾値
    """
    if method == 'percentile':
        # パーセンタイルベースの閾値
        percentile = kwargs.get('percentile', 95)
        return np.percentile(normal_scores, percentile)
    
    elif method == 'zscore':
        # Z-scoreによる外れ値検出（平均からn標準偏差）
        n_sigma = kwargs.get('n_sigma', 2.0)
        mean = np.mean(normal_scores)
        std = np.std(normal_scores)
        return mean + n_sigma * std
    
    elif method == 'iqr':
        # 四分位範囲による外れ値検出
        q1 = np.percentile(normal_scores, 25)
        q3 = np.percentile(normal_scores, 75)
        iqr = q3 - q1
        k = kwargs.get('k', 1.5)
        return q3 + k * iqr
    
    elif method == 'gmm':
        # ガウス混合モデルによる閾値決定
        from sklearn.mixture import GaussianMixture
        n_components = kwargs.get('n_components', 2)
        
        # スコアの形状を変更
        scores_reshaped = normal_scores.reshape(-1, 1)
        
        # GMMをフィット
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(scores_reshaped)
        
        # コンポーネントの平均と標準偏差
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        # 通常、大きい平均を持つコンポーネントが異常を表す
        normal_idx = np.argmin(means)
        abnormal_idx = np.argmax(means)
        
        # 異常クラスの平均 - n×標準偏差を閾値とする
        n_sigma = kwargs.get('n_sigma', 2.0)
        threshold = means[abnormal_idx] - n_sigma * np.sqrt(variances[abnormal_idx])
        
        return threshold
    
    else:
        # デフォルト: パーセンタイル
        return np.percentile(normal_scores, 95)

def normalize_scores(scores):
    """スコアの正規化（0になる問題の解決）"""
    epsilon = 1e-10  # 数値的安定性のための小さな値
    
    # Min-Max正規化
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # スコアの差が非常に小さい場合
    if max_score - min_score < epsilon:
        print("警告: スコアの変動が非常に小さいです。代替正規化を適用します。")
        # 平均を0、分散を1にする標準化
        mean = np.mean(scores)
        std = np.std(scores)
        
        # 標準偏差が非常に小さい場合はデフォルト値を使用
        if std < epsilon:
            std = 1.0
            print("警告: 標準偏差が非常に小さいです。デフォルト値を使用します。")
        
        normalized_scores = (scores - mean) / std
    else:
        # 通常のMin-Max正規化
        normalized_scores = (scores - min_score) / (max_score - min_score)
    
    return normalized_scores

def improved_anomaly_detection(train_folder='./data/train', test_folder='./data/test', 
                               threshold=None, threshold_method='percentile', use_pca=True):
    """改良版異常検知を実行する関数"""
    # モデルのロードと評価モードへの設定
    model = ImprovedViTFeatureExtractor().to(device)
    model.eval()
    
    # 正常画像の読み込み（データ拡張あり）
    print("正常画像を読み込んでいます...")
    train_imgs, _ = load_images_from_folder(train_folder, transform=train_transform, 
                                           augment=True, num_augmentations=4)
    
    if len(train_imgs) == 0:
        print("エラー: 正常画像が読み込めませんでした。パスを確認してください。")
        return
    
    print(f"データ拡張後の訓練画像数: {len(train_imgs)}")
    
    # 正常画像の特徴量を計算
    print("正常画像から特徴量を抽出しています...")
    train_features = []
    batch_size = 16  # バッチ処理で効率化
    
    with torch.no_grad():
        for i in range(0, len(train_imgs), batch_size):
            batch = train_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            train_features.append(feat.cpu().numpy())
    
    train_features = np.vstack(train_features)
    
    # 次元削減（オプション）
    if use_pca:
        print("PCAで特徴量の次元を削減しています...")
        pca = PCA(n_components=0.95)  # 95%の分散を説明する成分を保持
        train_features = pca.fit_transform(train_features)
        print(f"PCA後の特徴量の次元: {train_features.shape[1]}")
    
    # データが少なすぎる場合はkNNを使用、そうでない場合はロバスト共分散推定を試行
    use_knn = len(train_features) < 200
    
    if use_knn:
        print(f"データサイズが小さいため（{len(train_features)}個）、kNN法を使用します")
        nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_model.fit(train_features)
    else:
        try:
            # マハラノビス距離のためのロバスト共分散推定
            print("ロバスト共分散推定を計算しています...")
            # データポイント数に応じてサポート率を調整
            support_fraction = min(0.6, (len(train_features) - 1) / len(train_features))
            robust_cov = MinCovDet(support_fraction=support_fraction).fit(train_features)
        except ValueError as e:
            print(f"MinCovDetエラー: {e}")
            print("代替としてkNN法を使用します")
            use_knn = True
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(train_features)
    
    # 正常サンプルの距離を計算（閾値決定用）
    if use_knn:
        distances, _ = nn_model.kneighbors(train_features)
        normal_distances = [dist[0] for dist in distances]
    else:
        normal_distances = robust_cov.mahalanobis(train_features)
    
    # テスト画像の読み込みと特徴抽出
    print("テスト画像を評価しています...")
    test_imgs, filenames = load_images_from_folder(test_folder, transform=test_transform)
    
    if len(test_imgs) == 0:
        print("エラー: テスト画像が読み込めませんでした。パスを確認してください。")
        return
    
    test_features = []
    with torch.no_grad():
        for i in range(0, len(test_imgs), batch_size):
            batch = test_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            test_features.append(feat.cpu().numpy())
    
    test_features = np.vstack(test_features)
    
    # PCAで同じ部分空間に射影（使用した場合）
    if use_pca:
        test_features = pca.transform(test_features)
    
    # 距離計算
    if use_knn:
        distances, _ = nn_model.kneighbors(test_features)
        anomaly_scores = [dist[0] for dist in distances]
    else:
        # マハラノビス距離の計算
        anomaly_scores = robust_cov.mahalanobis(test_features)
    
    # スコアの正規化（閾値が0.00になる問題の対策）
    anomaly_scores = normalize_scores(anomaly_scores)
    
    # 閾値決定
    if threshold is None:
        # 正常データのスコアも正規化
        normal_scores = normalize_scores(normal_distances)
        # 適応的閾値計算
        threshold = compute_adaptive_threshold(normal_scores, method=threshold_method)
    
    # 結果の表示
    print(f"\n閾値: {threshold:.4f}\n")
    print("異常検知結果:")
    
    anomalies = []
    for filename, score in zip(filenames, anomaly_scores):
        result = "異常" if score > threshold else "正常"
        anomalies.append(result == "異常")
        print(f"{filename}: スコア = {score:.4f} -> {result}")
    
    # 異常スコアをプロット
    plt.figure(figsize=(12, 6))
    
    # ソートしてプロット
    sorted_indices = np.argsort(anomaly_scores)
    sorted_scores = [anomaly_scores[i] for i in sorted_indices]
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_anomalies = [anomalies[i] for i in sorted_indices]
    
    # 棒グラフの色を異常/正常で分ける
    colors = ['red' if is_anomaly else 'green' for is_anomaly in sorted_anomalies]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color=colors)
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'閾値 ({threshold:.4f})')
    
    plt.xticks(range(len(sorted_scores)), sorted_filenames, rotation=45, ha='right')
    plt.ylabel("異常スコア")
    plt.title("改良型ViTによる異常検知スコア")
    plt.legend()
    plt.tight_layout()
    
    # 正常と異常の画像数をカウント
    normal_count = anomalies.count(False)
    anomaly_count = anomalies.count(True)
    
    print(f"\n検出結果: 正常画像 {normal_count}枚, 異常画像 {anomaly_count}枚")
    
    plt.savefig('improved_anomaly_detection_results.png')
    plt.show()
    
    return anomaly_scores, threshold, filenames

def ensemble_anomaly_detection(train_folder='./data/train', test_folder='./data/test'):
    """
    複数のViTモデルをアンサンブルして精度を向上させる関数
    """
    # 使用するモデル設定（データが少ない場合は設定を減らす）
    model_configs = [
        {'name': 'vit_base_patch16_224', 'use_pca': True, 'method': 'knn'},
        {'name': 'vit_base_patch16_224', 'use_pca': False, 'method': 'knn'}
    ]
    
    # テスト画像の読み込み（ファイル名の記録用）
    test_imgs, filenames = load_images_from_folder(test_folder, transform=test_transform)
    
    if len(test_imgs) == 0:
        print("エラー: テスト画像が読み込めませんでした。パスを確認してください。")
        return
    
    # 各モデルの異常スコアを計算
    all_scores = []
    
    for i, config in enumerate(model_configs):
        print(f"\nモデル {i+1}/{len(model_configs)}: {config['name']} で異常検知を実行中...")
        
        # モデルのロードと設定
        model = ImprovedViTFeatureExtractor(model_name=config['name']).to(device)
        model.eval()
        
        # 正常画像の読み込み（データ拡張あり）
        train_imgs, _ = load_images_from_folder(train_folder, transform=train_transform, 
                                              augment=True, num_augmentations=4)
        
        # 正常画像の特徴量を計算
        train_features = []
        batch_size = 16
        
        with torch.no_grad():
            for j in range(0, len(train_imgs), batch_size):
                batch = train_imgs[j:j+batch_size].to(device)
                feat = model(batch)
                train_features.append(feat.cpu().numpy())
        
        train_features = np.vstack(train_features)
        
        # PCA（オプション）
        if config['use_pca']:
            pca = PCA(n_components=0.95)
            train_features = pca.fit_transform(train_features)
        
        # テスト画像の特徴抽出
        test_features = []
        with torch.no_grad():
            for j in range(0, len(test_imgs), batch_size):
                batch = test_imgs[j:j+batch_size].to(device)
                feat = model(batch)
                test_features.append(feat.cpu().numpy())
        
        test_features = np.vstack(test_features)
        
        # PCA（使用した場合）
        if config['use_pca']:
            test_features = pca.transform(test_features)
        
        # データが少なすぎる場合はkNNを使用、それ以外はMinCovDetを試行
        if config['method'] == 'knn' or len(train_features) < 200:
            print(f"データサイズが小さいため（{len(train_features)}個）、kNN法を使用します")
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(train_features)
            distances, _ = nn_model.kneighbors(test_features)
            scores = [dist[0] for dist in distances]
        else:
            try:
                # マハラノビス距離計算（support_fractionを調整）
                # データポイント数に応じてサポート率を調整
                support_fraction = min(0.6, (len(train_features) - 1) / len(train_features))
                robust_cov = MinCovDet(support_fraction=support_fraction).fit(train_features)
                scores = robust_cov.mahalanobis(test_features)
            except ValueError as e:
                print(f"MinCovDetエラー: {e}")
                print("代替としてkNN法を使用します")
                nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_model.fit(train_features)
                distances, _ = nn_model.kneighbors(test_features)
                scores = [dist[0] for dist in distances]
        
        # スコアの正規化
        scores = normalize_scores(scores)
        all_scores.append(scores)
    
    # アンサンブル（スコアの平均）
    ensemble_scores = np.mean(all_scores, axis=0)
    
    # 閾値計算（デフォルト: 0.5、スコアは0-1に正規化済み）
    ensemble_threshold = 0.5
    
    # 結果表示
    print("\nアンサンブル異常検知の結果:")
    anomalies = []
    for filename, score in zip(filenames, ensemble_scores):
        result = "異常" if score > ensemble_threshold else "正常"
        anomalies.append(result == "異常")
        print(f"{filename}: スコア = {score:.4f} -> {result}")
    
    # 可視化
    plt.figure(figsize=(12, 6))
    sorted_indices = np.argsort(ensemble_scores)
    sorted_scores = [ensemble_scores[i] for i in sorted_indices]
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_anomalies = [anomalies[i] for i in sorted_indices]
    
    colors = ['red' if is_anomaly else 'green' for is_anomaly in sorted_anomalies]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color=colors)
    plt.axhline(y=ensemble_threshold, color='orange', linestyle='--', label=f'閾値 ({ensemble_threshold:.4f})')
    
    plt.xticks(range(len(sorted_scores)), sorted_filenames, rotation=45, ha='right')
    plt.ylabel("異常スコア")
    plt.title("ViTアンサンブルによる異常検知スコア")
    plt.legend()
    plt.tight_layout()
    
    # 正常と異常の画像数をカウント
    normal_count = anomalies.count(False)
    anomaly_count = anomalies.count(True)
    
    print(f"\n検出結果: 正常画像 {normal_count}枚, 異常画像 {anomaly_count}枚")
    
    plt.savefig('ensemble_anomaly_detection_results.png')
    plt.show()
    
    return ensemble_scores, ensemble_threshold, filenames

# 元の異常検知関数（比較用に残す）
def original_anomaly_detection(train_folder='./data/train', test_folder='./data/test', threshold=None):
    """元の異常検知を実行する関数"""
    # モデルのロードと評価モードへの設定
    model = ViTFeatureExtractor().to(device)
    model.eval()
    
    # 正常画像の読み込み
    print("正常画像を読み込んでいます...")
    train_imgs, _ = load_images_from_folder(train_folder, transform=test_transform)
    
    if len(train_imgs) == 0:
        print("エラー: 正常画像が読み込めませんでした。パスを確認してください。")
        return
    
    # 正常画像の特徴量を計算
    print("正常画像から特徴量を抽出しています...")
    train_features = []
    batch_size = 16  # バッチ処理で効率化
    
    with torch.no_grad():
        for i in range(0, len(train_imgs), batch_size):
            batch = train_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            train_features.append(feat.cpu().numpy())
    
    train_features = np.vstack(train_features)
    
    # kNN で正常クラスタを構築
    print("kNNモデルを構築しています...")
    nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_model.fit(train_features)
    
    # テスト画像の読み込みと特徴抽出
    print("テスト画像を評価しています...")
    test_imgs, filenames = load_images_from_folder(test_folder, transform=test_transform)
    
    if len(test_imgs) == 0:
        print("エラー: テスト画像が読み込めませんでした。パスを確認してください。")
        return
    
    anomaly_scores = []
    with torch.no_grad():
        for i in range(0, len(test_imgs), batch_size):
            batch = test_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            feat_np = feat.cpu().numpy()
            
            distances, _ = nn_model.kneighbors(feat_np)
            anomaly_scores.extend([dist[0] for dist in distances])
    
    # 異常判定の閾値を決定（もし指定されていなければ）
    if threshold is None:
        # 正常データの距離の分布から閾値を決定（例: 平均 + 2*標準偏差）
        with torch.no_grad():
            normal_distances = []
            for i in range(0, len(train_imgs), batch_size):
                batch = train_imgs[i:i+batch_size].to(device)
                feat = model(batch)
                feat_np = feat.cpu().numpy()
                
                distances, _ = nn_model.kneighbors(feat_np)
                normal_distances.extend([dist[0] for dist in distances])
            
            mean_dist = np.mean(normal_distances)
            std_dist = np.std(normal_distances)
            threshold = mean_dist + 2 * std_dist
    
    # 結果の表示
    print(f"\n閾値: {threshold:.4f}\n")
    print("異常検知結果:")
    
    anomalies = []
    for filename, score in zip(filenames, anomaly_scores):
        result = "異常" if score > threshold else "正常"
        anomalies.append(result == "異常")
        print(f"{filename}: スコア = {score:.4f} -> {result}")
    
    # 異常スコアをプロット
    plt.figure(figsize=(12, 6))
    
    # ソートしてプロット
    sorted_indices = np.argsort(anomaly_scores)
    sorted_scores = [anomaly_scores[i] for i in sorted_indices]
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_anomalies = [anomalies[i] for i in sorted_indices]
    
    # 棒グラフの色を異常/正常で分ける
    colors = ['red' if is_anomaly else 'green' for is_anomaly in sorted_anomalies]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color=colors)
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'閾値 ({threshold:.4f})')
    
    plt.xticks(range(len(sorted_scores)), sorted_filenames, rotation=45, ha='right')
    plt.ylabel("異常スコア")
    plt.title("ViTによる異常検知スコア (オリジナル)")
    plt.legend()
    plt.tight_layout()
    
    # 正常と異常の画像数をカウント
    normal_count = anomalies.count(False)
    anomaly_count = anomalies.count(True)
    
    print(f"\n検出結果: 正常画像 {normal_count}枚, 異常画像 {anomaly_count}枚")
    
    plt.savefig('original_anomaly_detection_results.png')
    plt.show()
    
    return anomaly_scores, threshold, filenames

# ViTモデルから特徴抽出（元の実装、比較用）
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Identity()  # 出力を特徴ベクトルに
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='改良版ViTを使用した異常検知')
    parser.add_argument('--train', type=str, default='./data/train', help='正常画像のフォルダパス')
    parser.add_argument('--test', type=str, default='./data/test', help='テスト画像のフォルダパス')
    parser.add_argument('--method', type=str, default='improved',
                        choices=['original', 'improved', 'ensemble'],
                        help='使用する異常検知手法')
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        choices=['percentile', 'zscore', 'iqr', 'gmm'],
                        help='閾値決定方法')
    parser.add_argument('--threshold', type=float, default=None, help='異常判定の閾値（指定なしの場合は自動計算）')
    
    args = parser.parse_args()
    
    # 選択した方法で異常検知を実行
    if args.method == 'original':
        original_anomaly_detection(args.train, args.test, args.threshold)
    elif args.method == 'improved':
        improved_anomaly_detection(args.train, args.test, args.threshold, args.threshold_method)
    elif args.method == 'ensemble':
        ensemble_anomaly_detection(args.train, args.test)