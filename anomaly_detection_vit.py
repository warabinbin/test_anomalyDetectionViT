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

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Yu Gothic'

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNetの正規化パラメータ
])

def load_images_from_folder(folder):
    """フォルダから画像を読み込む関数"""
    images = []
    filenames = []
    
    if not os.path.exists(folder):
        print(f"Error: フォルダ '{folder}' が存在しません。")
        return torch.tensor([]), []
    
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not files:
        print(f"Warning: フォルダ '{folder}' に画像ファイルが見つかりません。")
        return torch.tensor([]), []
    
    for filename in files:
        try:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('RGB')
            img = transform(img)
            images.append(img)
            filenames.append(filename)
        except Exception as e:
            print(f"Warning: 画像 '{filename}' の読み込みに失敗しました: {str(e)}")
    
    if not images:
        return torch.tensor([]), []
    
    return torch.stack(images), filenames

# ViTモデルから特徴抽出
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Identity()  # 出力を特徴ベクトルに
    
    def forward(self, x):
        return self.model(x)

def anomaly_detection(train_folder='./data/train', test_folder='./data/test', threshold=None):
    """異常検知を実行する関数"""
    # モデルのロードと評価モードへの設定
    model = ViTFeatureExtractor().to(device)
    model.eval()
    
    # 正常画像の読み込み
    print("正常画像を読み込んでいます...")
    train_imgs, _ = load_images_from_folder(train_folder)
    
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
    test_imgs, filenames = load_images_from_folder(test_folder)
    
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
    plt.title("ViTによる異常検知スコア")
    plt.legend()
    plt.tight_layout()
    
    # 正常と異常の画像数をカウント
    normal_count = anomalies.count(False)
    anomaly_count = anomalies.count(True)
    
    print(f"\n検出結果: 正常画像 {normal_count}枚, 異常画像 {anomaly_count}枚")
    
    plt.savefig('anomaly_detection_results.png')
    plt.show()
    
    return anomaly_scores, threshold, filenames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ViTを使用した異常検知')
    parser.add_argument('--train', type=str, default='./data/train', help='正常画像のフォルダパス')
    parser.add_argument('--test', type=str, default='./data/test', help='テスト画像のフォルダパス')
    parser.add_argument('--threshold', type=float, default=None, help='異常判定の閾値（指定なしの場合は自動計算）')
    
    args = parser.parse_args()
    
    # 異常検知の実行
    anomaly_detection(args.train, args.test, args.threshold)
