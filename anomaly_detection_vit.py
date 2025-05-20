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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report, precision_recall_curve, average_precision_score
import seaborn as sns

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
    
    for filename in tqdm(files, desc=f"画像読み込み ({folder})"):
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
class ViTFeatureExtractor(nn.Module):
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

def plot_confusion_matrix(y_true, y_pred, title='混同行列'):
    """混同行列を描画する関数"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '異常'], 
                yticklabels=['正常', '異常'])
    plt.xlabel('予測値')
    plt.ylabel('真値')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_score, title='ROC曲線'):
    """ROC曲線を描画する関数"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('偽陽性率 (FPR)')
    plt.ylabel('真陽性率 (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

def plot_precision_recall_curve(y_true, y_score, title='適合率-再現率曲線'):
    """適合率-再現率曲線を描画する関数"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', label=f'AP = {ap:.4f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('再現率')
    plt.ylabel('適合率')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.show()

def evaluate_anomaly_detection(y_true, y_pred, y_scores):
    """異常検知の精度評価を行う関数"""
    # 分類レポート
    print("\n分類レポート:")
    print(classification_report(y_true, y_pred, target_names=['正常', '異常']))
    
    # 主要指標
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    
    print(f"精度 (Accuracy): {accuracy:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall): {recall:.4f}")
    print(f"F1スコア: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 混同行列
    plot_confusion_matrix(y_true, y_pred)
    
    # ROC曲線
    plot_roc_curve(y_true, y_scores)
    
    # 適合率-再現率曲線
    plot_precision_recall_curve(y_true, y_scores)
    
    # 評価指標の辞書を返す
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def anomaly_detection(train_folder='./data/train/normal', 
                      test_normal_folder='./data/test/normal',
                      test_abnormal_folder='./data/test/abnormal',
                      threshold=None, 
                      threshold_method='percentile',
                      model_name='vit_base_patch16_224', 
                      use_pca=True):
    """改良版異常検知を実行する関数（フォルダ構造対応・精度評価追加）"""
    print("\n====== 異常検知を開始します ======")
    print(f"使用モデル: {model_name}")
    print(f"PCA使用: {use_pca}")
    print(f"閾値決定法: {threshold_method}")
    
    # モデルのロードと評価モードへの設定
    model = ViTFeatureExtractor(model_name=model_name).to(device)
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
    
    # テスト用の正常画像の読み込みと特徴抽出
    print("テスト用正常画像を評価しています...")
    test_normal_imgs, normal_filenames = load_images_from_folder(test_normal_folder, transform=test_transform)
    
    if len(test_normal_imgs) == 0:
        print("エラー: テスト用正常画像が読み込めませんでした。パスを確認してください。")
        return
    
    # テスト用の異常画像の読み込みと特徴抽出
    print("テスト用異常画像を評価しています...")
    test_abnormal_imgs, abnormal_filenames = load_images_from_folder(test_abnormal_folder, transform=test_transform)
    
    if len(test_abnormal_imgs) == 0:
        print("エラー: テスト用異常画像が読み込めませんでした。パスを確認してください。")
        return
    
    # 全テスト画像を結合
    all_test_imgs = torch.cat([test_normal_imgs, test_abnormal_imgs], dim=0)
    all_filenames = normal_filenames + abnormal_filenames
    
    # 真のラベル（0: 正常, 1: 異常）
    y_true = np.array([0] * len(normal_filenames) + [1] * len(abnormal_filenames))
    
    # 特徴抽出
    print("全テスト画像から特徴量を抽出しています...")
    test_features = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(all_test_imgs), batch_size):
            batch = all_test_imgs[i:i+batch_size].to(device)
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
    
    # 異常判定（0: 正常, 1: 異常）
    y_pred = np.array([1 if score > threshold else 0 for score in anomaly_scores])
    
    # 結果の表示
    print(f"\n閾値: {threshold:.4f}\n")
    print("異常検知結果:")
    
    for filename, score, true_label, pred_label in zip(all_filenames, anomaly_scores, y_true, y_pred):
        true_status = "異常" if true_label == 1 else "正常"
        pred_status = "異常" if pred_label == 1 else "正常"
        correct = "○" if true_label == pred_label else "×"
        print(f"{filename}: スコア = {score:.4f}, 真値 = {true_status}, 予測 = {pred_status} {correct}")
    
    # 異常スコアをプロット
    plt.figure(figsize=(12, 6))
    
    # 正常と異常でグループ化してプロット
    normal_scores = anomaly_scores[:len(normal_filenames)]
    abnormal_scores = anomaly_scores[len(normal_filenames):]
    
    plt.hist(normal_scores, bins=20, alpha=0.7, label='正常', color='green')
    plt.hist(abnormal_scores, bins=20, alpha=0.7, label='異常', color='red')
    plt.axvline(x=threshold, color='orange', linestyle='--', label=f'閾値 ({threshold:.4f})')
    
    plt.xlabel("異常スコア")
    plt.ylabel("頻度")
    plt.title("異常スコア分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig('anomaly_score_distribution.png')
    plt.show()
    
    # 散布図（正常と異常を区別）
    if use_pca and test_features.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(test_features[:len(normal_filenames), 0], 
                   test_features[:len(normal_filenames), 1], 
                   c='green', marker='o', alpha=0.7, label='正常')
        plt.scatter(test_features[len(normal_filenames):, 0], 
                   test_features[len(normal_filenames):, 1], 
                   c='red', marker='x', alpha=0.7, label='異常')
        plt.title('PCA特徴空間における正常・異常サンプルの分布')
        plt.xlabel('第1主成分')
        plt.ylabel('第2主成分')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('feature_space_distribution.png')
        plt.show()
    
    # 精度評価
    print("\n====== 精度評価 ======")
    metrics = evaluate_anomaly_detection(y_true, y_pred, anomaly_scores)
    
    return {
        'anomaly_scores': anomaly_scores,
        'threshold': threshold,
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='改良版ViTを使用した異常検知')
    parser.add_argument('--train', type=str, default='./data/train/normal', help='正常画像のフォルダパス')
    parser.add_argument('--test_normal', type=str, default='./data/test/normal', help='テスト用正常画像のフォルダパス')
    parser.add_argument('--test_abnormal', type=str, default='./data/test/abnormal', help='テスト用異常画像のフォルダパス')
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        choices=['percentile', 'zscore', 'iqr', 'gmm'],
                        help='閾値決定方法')
    parser.add_argument('--threshold', type=float, default=None, help='異常判定の閾値（指定なしの場合は自動計算）')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='使用するViTモデル')
    parser.add_argument('--use_pca', action='store_true', help='PCAによる次元削減を使用する')
    
    args = parser.parse_args()
    
    # 異常検知を実行
    results = anomaly_detection(
        train_folder=args.train,
        test_normal_folder=args.test_normal,
        test_abnormal_folder=args.test_abnormal,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        model_name=args.model,
        use_pca=args.use_pca
    )