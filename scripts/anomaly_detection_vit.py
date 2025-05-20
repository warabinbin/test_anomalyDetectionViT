import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import copy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report, precision_recall_curve, average_precision_score
)

class Config:
    """設定クラス - すべての設定パラメータを一箇所に集約"""
    def __init__(self):
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 画像変換設定
        self.image_size = (224, 224)
        self.norm_mean = [0.485, 0.456, 0.406]  # ImageNet平均
        self.norm_std = [0.229, 0.224, 0.225]   # ImageNet標準偏差
        
        # モデル設定
        self.model_name = 'vit_base_patch16_224'
        self.batch_size = 16
        self.feature_extraction_method = 'enhanced'  # 'original', 'enhanced', 'autoencoder'
        
        # 異常検知設定
        self.use_pca = True
        self.pca_variance = 0.95
        self.max_pca_dim = 100  # PCAの最大次元数
        self.knn_threshold = 200  # データ数がこれ以下の場合はkNNを使用
        
        # データパス設定
        self.train_folder = './data/train/normal'
        self.test_normal_folder = './data/test/normal'
        self.test_abnormal_folder = './data/test/abnormal'
        
        # 閾値設定
        self.threshold = None
        self.threshold_method = 'balanced_accuracy'  # 'percentile', 'balanced_accuracy', 'f1', 'pr_auc'
        
        # 出力設定
        self.output_dir = './output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 改善のための新しい設定
        self.use_ensemble = True  # アンサンブル検出を使用
        self.use_synthetic_samples = True  # 合成異常サンプルを使用
        self.synthetic_sample_count = 200  # 合成サンプル数
        self.use_data_augmentation = True  # データ拡張を使用
        self.visualization = True  # t-SNEなどの可視化を行う
        self.memory_efficient = True  # メモリ効率を優先

    def get_transform(self, train=False):
        """画像変換パイプラインを返す"""
        if train and self.use_data_augmentation:
            # 訓練データには強力な拡張を適用
            return transforms.Compose([
                transforms.Resize((self.image_size[0] + 20, self.image_size[1] + 20)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ])
        else:
            # テストデータには最小限の前処理のみ
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ])
    
    def __str__(self):
        """設定の文字列表現"""
        return (
            f"Device: {self.device}\n"
            f"Model: {self.model_name}\n"
            f"Feature extraction: {self.feature_extraction_method}\n"
            f"Use PCA: {self.use_pca}\n"
            f"Max PCA dim: {self.max_pca_dim}\n"
            f"Threshold method: {self.threshold_method}\n"
            f"Batch size: {self.batch_size}\n"
            f"Use ensemble: {self.use_ensemble}\n"
            f"Use synthetic samples: {self.use_synthetic_samples}\n"
            f"Use data augmentation: {self.use_data_augmentation}\n"
            f"Memory efficient: {self.memory_efficient}"
        )


class ViTFeatureExtractor(nn.Module):
    """Vision Transformerから特徴を抽出するモデル（オリジナル）"""
    def __init__(self, model_name='vit_base_patch16_224'):
        """
        Args:
            model_name: 使用するViTモデル名
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Identity()  # 出力層を特徴ベクトルに
        
        # 中間層にアクセスするためのフック
        self.hook_features = []
        self.hooks = []
        
        # 最後から3つのTransformerブロックにフックを追加
        for block in self.model.blocks[-3:]:
            self.hooks.append(block.register_forward_hook(self._get_features))
    
    def _get_features(self, module, input, output):
        """フックの関数: 中間層の特徴を保存"""
        self.hook_features.append(output[:, 0])  # CLSトークンを保存
    
    def forward(self, x):
        """
        順伝播関数
        
        Args:
            x: 入力テンソル
            
        Returns:
            特徴ベクトル
        """
        self.hook_features = []
        feat = self.model(x)
        
        # 中間層のCLSトークンを連結
        if self.hook_features:
            all_features = torch.cat([feat] + self.hook_features, dim=1)
            return all_features
        
        return feat
    
    def __del__(self):
        """フックを適切に解除"""
        for hook in self.hooks:
            hook.remove()


class EnhancedFeatureExtractor(nn.Module):
    """改善版特徴抽出器 - 多様な特徴表現を抽出"""
    def __init__(self, model_name='vit_base_patch16_224', memory_efficient=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Identity()
        self.memory_efficient = memory_efficient
        
        # 特徴抽出のためのフック
        self.hook_features = []
        self.hooks = []
        
        if memory_efficient:
            # メモリ効率モード: 最後のブロックのみを使用
            block = self.model.blocks[-1]
            self.hooks.append(block.register_forward_hook(self._get_simplified_features))
        else:
            # 異なる層からの特徴を取得（浅い層と深い層の両方）
            layers_to_hook = [0, 6, -1]  # 最初、中間、最後の層
            for i in layers_to_hook:
                if i >= 0 and i < len(self.model.blocks):
                    block = self.model.blocks[i]
                else:
                    block = self.model.blocks[i]
                self.hooks.append(block.register_forward_hook(self._get_features))
        
    def _get_features(self, module, input, output):
        """多様な特徴を抽出"""
        # CLSトークン
        cls_token = output[:, 0]
        
        # パッチトークンの統計量
        patch_tokens = output[:, 1:]
        mean_patch = torch.mean(patch_tokens, dim=1)
        max_patch = torch.max(patch_tokens, dim=1).values
        std_patch = torch.std(patch_tokens, dim=1)
        
        # 特徴を結合
        combined = torch.cat([
            cls_token, mean_patch, max_patch, std_patch
        ], dim=1)
        
        self.hook_features.append(combined)
    
    def _get_simplified_features(self, module, input, output):
        """よりコンパクトな特徴を抽出（メモリ効率モード）"""
        # CLSトークンのみ
        cls_token = output[:, 0]
        
        # パッチ統計情報（最小限）
        patch_tokens = output[:, 1:]
        mean_patch = torch.mean(patch_tokens, dim=1)
        
        # 特徴を結合（より少ない次元）
        self.hook_features.append(torch.cat([cls_token, mean_patch], dim=1))
    
    def forward(self, x):
        self.hook_features = []
        feat = self.model(x)
        
        if not self.hook_features:
            return feat  # バックアップとして
        
        # 異なる層の特徴を連結
        multi_scale = torch.cat(self.hook_features, dim=1)
        return multi_scale
    
    def __del__(self):
        """フックを適切に解除"""
        for hook in self.hooks:
            hook.remove()


class ViTAutoencoder(nn.Module):
    """ViTベースのオートエンコーダ - 再構成と特徴抽出の両方を行う"""
    def __init__(self, model_name='vit_base_patch16_224', img_size=224, memory_efficient=False):
        super().__init__()
        # エンコーダとしてViTを使用
        self.encoder = timm.create_model(model_name, pretrained=True)
        self.feature_dim = self.encoder.head.in_features  # 特徴次元（通常は768）
        self.encoder.head = nn.Identity()  # 出力層を特徴ベクトルに
        self.memory_efficient = memory_efficient
        
        # 中間層のフック
        self.hook_features = []
        self.hooks = []
        
        # フックの追加（メモリ効率モードでは最後の層のみ）
        if memory_efficient:
            block = self.encoder.blocks[-1]
            self.hooks.append(block.register_forward_hook(self._get_features))
        else:
            # 最後から3つのTransformerブロックにフックを追加
            for block in self.encoder.blocks[-3:]:
                self.hooks.append(block.register_forward_hook(self._get_features))
        
        # デコーダの構築
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * img_size * img_size),
            nn.Sigmoid()  # 出力を[0,1]の範囲に正規化
        )
        
        self.img_size = img_size
    
    def _get_features(self, module, input, output):
        """フックの関数: 中間層の特徴を保存"""
        self.hook_features.append(output[:, 0])  # CLSトークンを保存
    
    def forward(self, x):
        """
        順伝播関数 - 特徴と再構成画像を返す
        
        Args:
            x: 入力画像テンソル
            
        Returns:
            (特徴ベクトル, 再構成画像)のタプル
        """
        self.hook_features = []
        features = self.encoder(x)
        
        # 中間層の特徴を結合
        if self.hook_features:
            all_features = torch.cat([features] + self.hook_features, dim=1)
        else:
            all_features = features
        
        # 再構成画像の生成
        reconstructed = self.decoder(features)
        reconstructed = reconstructed.view(-1, 3, self.img_size, self.img_size)
        
        return all_features, reconstructed
    
    def __del__(self):
        """フックを適切に解除"""
        for hook in self.hooks:
            hook.remove()


class ImageDataset:
    """画像データセットを管理するクラス"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.train_transform = config.get_transform(train=True)
        self.test_transform = config.get_transform(train=False)
    
    def load_images(self, folder, is_train=False):
        """
        フォルダから画像を読み込む
        
        Args:
            folder: 画像フォルダのパス
            is_train: 訓練データかどうか（データ拡張の適用）
            
        Returns:
            画像テンソルとファイル名のタプル
        """
        images = []
        filenames = []
        
        if not os.path.exists(folder):
            print(f"エラー: フォルダ '{folder}' が存在しません。")
            return torch.tensor([]), []
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        )]
        
        if not files:
            print(f"警告: フォルダ '{folder}' に画像ファイルが見つかりません。")
            return torch.tensor([]), []
        
        transform = self.train_transform if is_train else self.test_transform
        
        for filename in tqdm(files, desc=f"画像読み込み ({folder})"):
            try:
                path = os.path.join(folder, filename)
                img = Image.open(path).convert('RGB')
                
                transformed_img = transform(img)
                images.append(transformed_img)
                filenames.append(filename)
                
            except Exception as e:
                print(f"警告: 画像 '{filename}' の読み込みに失敗しました: {str(e)}")
        
        if not images:
            return torch.tensor([]), []
        
        return torch.stack(images), filenames
    
    def load_dataset(self):
        """
        訓練・テストデータセットを読み込む
        
        Returns:
            データセットの辞書
        """
        # 正常訓練データ
        train_imgs, _ = self.load_images(self.config.train_folder, is_train=True)
        if len(train_imgs) == 0:
            raise ValueError(f"正常訓練画像が読み込めませんでした。パス: {self.config.train_folder}")
        
        # 正常テストデータ
        test_normal_imgs, normal_filenames = self.load_images(self.config.test_normal_folder)
        if len(test_normal_imgs) == 0:
            raise ValueError(f"正常テスト画像が読み込めませんでした。パス: {self.config.test_normal_folder}")
        
        # 異常テストデータ
        test_abnormal_imgs, abnormal_filenames = self.load_images(self.config.test_abnormal_folder)
        if len(test_abnormal_imgs) == 0:
            raise ValueError(f"異常テスト画像が読み込めませんでした。パス: {self.config.test_abnormal_folder}")
        
        # 全テスト画像を結合
        all_test_imgs = torch.cat([test_normal_imgs, test_abnormal_imgs], dim=0)
        all_filenames = normal_filenames + abnormal_filenames
        
        # 真のラベル（0: 正常, 1: 異常）
        y_true = np.array([0] * len(normal_filenames) + [1] * len(abnormal_filenames))
        
        return {
            'train_imgs': train_imgs,
            'test_normal_imgs': test_normal_imgs,
            'test_abnormal_imgs': test_abnormal_imgs,
            'all_test_imgs': all_test_imgs,
            'normal_filenames': normal_filenames,
            'abnormal_filenames': abnormal_filenames,
            'all_filenames': all_filenames,
            'y_true': y_true
        }


class FeatureExtractor:
    """特徴抽出を行うクラス"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        
        # 特徴抽出方法に基づいてモデルを初期化
        if config.feature_extraction_method == 'enhanced':
            self.model = EnhancedFeatureExtractor(
                model_name=config.model_name, 
                memory_efficient=config.memory_efficient
            ).to(config.device)
        elif config.feature_extraction_method == 'autoencoder':
            self.model = ViTAutoencoder(
                model_name=config.model_name, 
                img_size=config.image_size[0],
                memory_efficient=config.memory_efficient
            ).to(config.device)
        else:
            self.model = ViTFeatureExtractor(model_name=config.model_name).to(config.device)
            
        self.model.eval()
        
    def extract_features(self, images):
        """
        画像テンソルから特徴量を抽出
        
        Args:
            images: 画像テンソル
            
        Returns:
            特徴量のNumPy配列とオプションで再構成画像
        """
        features = []
        reconstructed_images = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.config.device)
                
                # オートエンコーダを使用する場合は特徴と再構成画像を取得
                if self.config.feature_extraction_method == 'autoencoder':
                    feat, recon = self.model(batch)
                    features.append(feat.cpu().numpy())
                    reconstructed_images.append(recon.cpu())
                else:
                    feat = self.model(batch)
                    features.append(feat.cpu().numpy())
                
                # メモリ解放
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if reconstructed_images:
            return np.vstack(features), torch.cat(reconstructed_images)
        else:
            return np.vstack(features)


class AnomalyDetector:
    """異常検知を行うクラス（オリジナル）"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.pca = None
        self.nn_model = None
        self.robust_cov = None
        self.use_knn = False
    
    def fit(self, train_features):
        """
        正常データでモデルを訓練
        
        Args:
            train_features: 訓練用特徴量
            
        Returns:
            self
        """
        # PCAによる次元削減（オプション）
        if self.config.use_pca:
            print("PCAで特徴量の次元を削減しています...")
            # 最大次元数を制限
            max_components = min(self.config.max_pca_dim, train_features.shape[0], train_features.shape[1])
            self.pca = PCA(n_components=max_components)
            train_features = self.pca.fit_transform(train_features)
            print(f"PCA後の特徴量の次元: {train_features.shape[1]}")
        
        # データが少なすぎる場合はkNNを使用、そうでない場合はロバスト共分散推定を試行
        self.use_knn = len(train_features) < self.config.knn_threshold
        
        if self.use_knn:
            print(f"データサイズが小さいため（{len(train_features)}個）、kNN法を使用します")
            self.nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            self.nn_model.fit(train_features)
        else:
            try:
                # マハラノビス距離のためのロバスト共分散推定
                print("ロバスト共分散推定を計算しています...")
                # データポイント数に応じてサポート率を調整
                support_fraction = min(0.6, (len(train_features) - 1) / len(train_features))
                self.robust_cov = MinCovDet(support_fraction=support_fraction).fit(train_features)
            except (ValueError, MemoryError) as e:
                print(f"MinCovDetエラー: {e}")
                print("代替としてkNN法を使用します")
                self.use_knn = True
                self.nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
                self.nn_model.fit(train_features)
        
        # 正常サンプルの距離を計算（閾値決定用）- PCA変換はもう不要
        if self.use_knn:
            distances, _ = self.nn_model.kneighbors(train_features)
            return [dist[0] for dist in distances]
        else:
            return self.robust_cov.mahalanobis(train_features)
    
    def compute_distances(self, features):
        """
        特徴量から異常スコア（距離）を計算
        
        Args:
            features: 特徴量
            
        Returns:
            距離のリスト
        """
        # 修正: PCAが使用されていれば、同じ変換を適用
        if self.config.use_pca and self.pca is not None:
            features = self.pca.transform(features)
            
        if self.use_knn:
            distances, _ = self.nn_model.kneighbors(features)
            return [dist[0] for dist in distances]
        else:
            return self.robust_cov.mahalanobis(features)
    
    def transform(self, features):
        """
        特徴量に次元削減を適用
        
        Args:
            features: 特徴量
            
        Returns:
            変換後の特徴量
        """
        if self.config.use_pca and self.pca is not None:
            return self.pca.transform(features)
        return features
    
    @staticmethod
    def normalize_scores(scores):
        """
        スコアの正規化
        
        Args:
            scores: 異常スコアのリスト
            
        Returns:
            正規化されたスコア
        """
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
    
    @staticmethod
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


class OptimizedThresholdSelector:
    """最適な閾値を選択するクラス"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        
    def find_optimal_threshold(self, y_true, anomaly_scores, method='f1'):
        """
        最適な閾値を見つける
        
        Args:
            y_true: 真のラベル
            anomaly_scores: 異常スコア
            method: 最適化指標 ('f1', 'balanced_accuracy', 'pr_auc')
            
        Returns:
            最適な閾値
        """
        # 候補となる閾値の範囲を生成
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        thresholds = np.linspace(min_score, max_score, 100)
        
        best_score = 0
        best_threshold = None
        
        # 各閾値でスコアを計算
        for threshold in thresholds:
            y_pred = np.array([1 if score > threshold else 0 for score in anomaly_scores])
            
            if method == 'f1':
                score = f1_score(y_true, y_pred)
            elif method == 'balanced_accuracy':
                # クラス不均衡を考慮した精度
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (specificity + sensitivity) / 2
            elif method == 'pr_auc':
                # PR曲線下面積
                precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
                score = auc(recall, precision)
            else:
                # デフォルトはF1スコア
                score = f1_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"最適な閾値: {best_threshold:.4f} ({method}スコア: {best_score:.4f})")
        return best_threshold


class DeepSVDD:
    """Deep Support Vector Data Description による異常検知"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.center = None
        self.radius = 0
        self.nu = 0.05  # 外れ値の割合を指定するパラメータ
        self.pca = None  # PCA変換用
    
    def fit(self, features):
        """
        Deep SVDDのハイパースフィアの中心と半径を計算
        
        Args:
            features: 訓練用特徴量
            
        Returns:
            異常スコア
        """
        # PCA変換を適用（必要な場合）
        if self.config.use_pca:
            max_components = min(self.config.max_pca_dim, features.shape[0], features.shape[1])
            self.pca = PCA(n_components=max_components)
            features = self.pca.fit_transform(features)

        self.center = np.mean(features, axis=0)
        
        # 各点から中心までの距離を計算
        distances = np.linalg.norm(features - self.center, axis=1)
        
        # 半径を設定（νパラメータで調整可能な割合のデータを含む）
        self.radius = np.quantile(distances, 1 - self.nu)
        
        return distances
    
    def compute_distances(self, features):
        """
        特徴量から異常スコア（距離）を計算
        
        Args:
            features: 特徴量
            
        Returns:
            距離のリスト
        """
        if self.center is None:
            raise ValueError("fit()メソッドを先に呼び出してください")
        
        # PCA変換を適用（必要な場合）
        if self.config.use_pca and self.pca is not None:
            features = self.pca.transform(features)
            
        return np.linalg.norm(features - self.center, axis=1)


class MemoryBankDetector:
    """メモリバンクベースの異常検知器"""
    def __init__(self, config, max_samples=1000):
        """
        Args:
            config: 設定オブジェクト
            max_samples: メモリバンクの最大サンプル数
        """
        self.config = config
        self.max_samples = max_samples
        self.memory_bank = None
        self.n_neighbors = 5  # k近傍の数
        self.pca = None  # PCA変換用
        
    def fit(self, features):
        """
        メモリバンクに特徴を保存
        
        Args:
            features: 訓練用特徴量
            
        Returns:
            異常スコア
        """
        # PCA変換を適用（必要な場合）
        if self.config.use_pca:
            max_components = min(self.config.max_pca_dim, features.shape[0], features.shape[1])
            self.pca = PCA(n_components=max_components)
            features = self.pca.fit_transform(features)
        
        if len(features) > self.max_samples:
            # ランダムサンプリングして保存サイズを制限
            indices = np.random.choice(len(features), self.max_samples, replace=False)
            self.memory_bank = features[indices]
        else:
            self.memory_bank = features
            
        # 各サンプルのメモリバンクとの最小距離を計算
        # k近傍の数（メモリバンクのサイズに合わせて調整）
        k = min(self.n_neighbors, len(self.memory_bank))
        
        # k近傍検索器の構築
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.memory_bank)
        
        # 各特徴量に対するk近傍距離を計算
        distances, _ = nn.kneighbors(features)
        
        # 平均k近傍距離を返す
        return np.mean(distances, axis=1)
    
    def compute_distances(self, features):
        """
        各サンプルのメモリバンク内のk個の最も近い要素との平均距離を計算
        
        Args:
            features: 特徴量
            
        Returns:
            距離のリスト
        """
        if self.memory_bank is None:
            raise ValueError("fit()メソッドを先に呼び出してください")
        
        # PCA変換を適用（必要な場合）
        if self.config.use_pca and self.pca is not None:
            features = self.pca.transform(features)
            
        # k近傍の数（メモリバンクのサイズに合わせて調整）
        k = min(self.n_neighbors, len(self.memory_bank))
        
        # k近傍検索器の構築
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.memory_bank)
        
        # 各特徴量に対するk近傍距離を計算
        distances, _ = nn.kneighbors(features)
        
        # 平均k近傍距離を返す
        return np.mean(distances, axis=1)


class RobustEnsembleDetector:
    """複数の検出器を組み合わせたアンサンブル検出器"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        
        # 複数の異常検出アルゴリズムを初期化（メモリ効率モードでは単純な検出器のみ）
        if config.memory_efficient:
            self.detectors = {
                'knn': NearestNeighbors(n_neighbors=5, metric='euclidean'),
                'deep_svdd': DeepSVDD(config)
            }
        else:
            self.detectors = {
                'mahalanobis': AnomalyDetector(config),
                'deep_svdd': DeepSVDD(config),
                'memory_bank': MemoryBankDetector(config)
            }
            
            # Isolation Forestを追加（メモリ効率モードでなければ）
            if len(config.train_folder) >= 50:  # 十分なデータがある場合
                self.detectors['isolation_forest'] = IsolationForest(
                    n_estimators=100, 
                    contamination=0.1,
                    random_state=42
                )
        
        self.pca = None
        self.weights = None
        self.is_fitted = False
    
    def fit(self, train_features, validation_data=None):
        """
        複数の検出器を学習し、最適な重みを計算
        
        Args:
            train_features: 訓練用特徴量
            validation_data: (特徴量, ラベル)の検証データタプル（オプション）
            
        Returns:
            異常スコア
        """
        # PCAによる次元削減（オプション）
        if self.config.use_pca:
            print("PCAで特徴量の次元を削減しています...")
            # 最大次元数を制限
            max_components = min(self.config.max_pca_dim, train_features.shape[0], train_features.shape[1])
            self.pca = PCA(n_components=max_components)
            train_features = self.pca.fit_transform(train_features)
            print(f"PCA後の特徴量の次元: {train_features.shape[1]}")
            
            # 検証データも変換
            if validation_data:
                val_features, val_labels = validation_data
                val_features = self.pca.transform(val_features)
                validation_data = (val_features, val_labels)
        
        # 各検出器の学習と訓練データのスコア計算
        scores_by_detector = {}
        failed_detectors = []  # 失敗した検出器のリスト
        
        for name, detector in list(self.detectors.items()):
            print(f"'{name}' 検出器を学習中...")
            
            try:
                if name in ['mahalanobis', 'deep_svdd', 'memory_bank']:
                    # 自作検出器 - PCAは既に適用済みなので、そのまま渡す
                    scores = detector.fit(train_features)
                    scores_by_detector[name] = scores
                else:
                    # scikit-learn検出器
                    detector.fit(train_features)
                    
                    if hasattr(detector, 'score_samples'):
                        scores = -detector.score_samples(train_features)
                    elif hasattr(detector, 'decision_function'):
                        scores = -detector.decision_function(train_features)
                    elif name == 'knn':
                        # NearestNeighborsはscore_samplesもdecision_functionも持たないのでkneighborsを使用
                        distances, _ = detector.kneighbors(train_features)
                        scores = np.mean(distances, axis=1)
                    else:
                        scores = np.zeros(len(train_features))
                        
                    scores_by_detector[name] = scores
            except Exception as e:
                print(f"警告: '{name}' 検出器の学習中にエラーが発生しました: {str(e)}")
                # 失敗した検出器をリストに追加
                failed_detectors.append(name)
        
        # 反復が終わった後に、失敗した検出器を削除
        for name in failed_detectors:
            if name in self.detectors:
                del self.detectors[name]
        
        # 検出器が全て失敗した場合
        if not self.detectors:
            raise ValueError("すべての検出器が学習に失敗しました。次元削減を強化するか、別の手法を試してください。")
        
        # 検証データがあれば、それを使って重みを最適化
        if validation_data:
            self.weights = self._optimize_weights(scores_by_detector, validation_data)
        else:
            # 検証データがなければ均等に重み付け
            self.weights = {name: 1.0/len(self.detectors) for name in self.detectors}
        
        print("検出器の重み:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        self.is_fitted = True
        return self._combine_scores(scores_by_detector)
    
    def compute_distances(self, features):
        """
        各検出器でスコアを計算し、重み付けして結合
        
        Args:
            features: 特徴量
            
        Returns:
            異常スコア
        """
        if not self.is_fitted:
            raise ValueError("compute_distances()を呼び出す前にfit()で学習する必要があります")
        
        # PCA変換
        if self.config.use_pca and self.pca is not None:
            features = self.pca.transform(features)
        
        # 各検出器でスコアを計算
        scores_by_detector = {}
        
        for name, detector in self.detectors.items():
            if name in ['mahalanobis', 'deep_svdd', 'memory_bank']:
                # 自作検出器
                scores = detector.compute_distances(features)
            else:
                # scikit-learn検出器
                if hasattr(detector, 'score_samples'):
                    scores = -detector.score_samples(features)
                elif hasattr(detector, 'decision_function'):
                    scores = -detector.decision_function(features)
                elif name == 'knn':
                    # NearestNeighborsはkneighborsを使用
                    distances, _ = detector.kneighbors(features)
                    scores = np.mean(distances, axis=1)
                else:
                    scores = np.zeros(len(features))
                    
            scores_by_detector[name] = scores
        
        # 結合スコアを計算
        return self._combine_scores(scores_by_detector)
    
    def _combine_scores(self, scores_by_detector):
        """
        各検出器のスコアを正規化して重み付け結合
        
        Args:
            scores_by_detector: 検出器別のスコア辞書
            
        Returns:
            結合スコア
        """
        # 検出器が1つもない場合
        if not scores_by_detector:
            return np.array([])
            
        normalized_scores = {}
        
        # 各検出器のスコアを正規化
        for name, scores in scores_by_detector.items():
            normalized_scores[name] = AnomalyDetector.normalize_scores(scores)
        
        # 重み付き合計を計算
        combined_scores = np.zeros(len(next(iter(normalized_scores.values()))))
        for name, scores in normalized_scores.items():
            if name in self.weights:
                combined_scores += self.weights[name] * scores
        
        return combined_scores
    
    def _optimize_weights(self, detector_scores, validation_data):
        """
        検証データを使って最適な重みを見つける
        
        Args:
            detector_scores: 訓練データに対する各検出器のスコア
            validation_data: (特徴量, ラベル)の検証データタプル
            
        Returns:
            各検出器の最適な重み
        """
        val_features, val_labels = validation_data
        weights = {}
        total_auc = 0
        
        for name, detector in self.detectors.items():
            try:
                # 検証データでスコアを計算
                if name in ['mahalanobis', 'deep_svdd', 'memory_bank']:
                    # 自作検出器
                    val_scores = detector.compute_distances(val_features)
                else:
                    # scikit-learn検出器
                    if hasattr(detector, 'score_samples'):
                        val_scores = -detector.score_samples(val_features)
                    elif hasattr(detector, 'decision_function'):
                        val_scores = -detector.decision_function(val_features)
                    elif name == 'knn':
                        # NearestNeighborsはkneighborsを使用
                        distances, _ = detector.kneighbors(val_features)
                        val_scores = np.mean(distances, axis=1)
                    else:
                        val_scores = np.zeros(len(val_features))
                
                # AUCスコアを計算
                auc_score = roc_auc_score(val_labels, val_scores)
                weights[name] = max(0.1, auc_score)  # 最低重みを0.1に設定
                total_auc += weights[name]
                
                print(f"  {name} 検出器のAUC: {auc_score:.4f}")
            
            except Exception as e:
                print(f"警告: '{name}' 検出器のAUC計算中にエラーが発生しました: {str(e)}")
                # 重みを0にして実質的に無効化
                weights[name] = 0
        
        # 重みの正規化（合計が1になるように）
        if total_auc > 0:
            for name in weights:
                weights[name] /= total_auc
        else:
            # AUCが計算できない場合は均等に
            weights = {name: 1.0/len(self.detectors) for name in self.detectors}
        
        return weights


class Evaluator:
    """異常検知の評価を行うクラス"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
    
    def evaluate(self, y_true, y_pred, anomaly_scores, all_filenames=None):
        """
        異常検知の性能を評価
        
        Args:
            y_true: 真のラベル
            y_pred: 予測ラベル
            anomaly_scores: 異常スコア
            all_filenames: ファイル名リスト（オプション）
            
        Returns:
            評価指標の辞書
        """
        # 分類レポート
        print("\n分類レポート:")
        print(classification_report(y_true, y_pred, target_names=['正常', '異常']))
        
        # 主要指標
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, anomaly_scores)
        except Exception as e:
            print(f"AUC計算エラー: {str(e)}")
            auc = 0.0
        
        print(f"精度 (Accuracy): {accuracy:.4f}")
        print(f"適合率 (Precision): {precision:.4f}")
        print(f"再現率 (Recall): {recall:.4f}")
        print(f"F1スコア: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # バランス精度も計算
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = (specificity + sensitivity) / 2
        print(f"バランス精度: {balanced_acc:.4f}")
        
        # 混同行列
        self._plot_confusion_matrix(y_true, y_pred)
        
        # ROC曲線
        self._plot_roc_curve(y_true, anomaly_scores)
        
        # 適合率-再現率曲線
        self._plot_precision_recall_curve(y_true, anomaly_scores)
        
        # 個別の予測結果
        if all_filenames:
            self._print_prediction_results(all_filenames, anomaly_scores, y_true, y_pred)
        
        # 評価指標の辞書を返す
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'balanced_accuracy': balanced_acc
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, title='混同行列'):
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
        
        # 出力ディレクトリに保存
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path)
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_score, title='ROC曲線'):
        """ROC曲線を描画する関数"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', label=f'AUC = {auc_score:.4f}')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('偽陽性率 (FPR)')
            plt.ylabel('真陽性率 (TPR)')
            plt.title(title)
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            
            # 出力ディレクトリに保存
            output_path = os.path.join(self.config.output_dir, 'roc_curve.png')
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"ROC曲線のプロット中にエラーが発生しました: {str(e)}")
    
    def _plot_precision_recall_curve(self, y_true, y_score, title='適合率-再現率曲線'):
        """適合率-再現率曲線を描画する関数"""
        try:
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
            
            # 出力ディレクトリに保存
            output_path = os.path.join(self.config.output_dir, 'precision_recall_curve.png')
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"適合率-再現率曲線のプロット中にエラーが発生しました: {str(e)}")
    
    def _print_prediction_results(self, all_filenames, anomaly_scores, y_true, y_pred):
        """個別の予測結果を表示する関数"""
        print("\n予測結果詳細:")
        # 最大で30件表示
        disp_count = min(30, len(all_filenames))
        
        # anomaly_scoresでソートして上位/下位のサンプルを表示
        indices = np.argsort(anomaly_scores)
        
        # 最も正常と判定されたサンプル
        print("\n最も正常と判定されたサンプル:")
        for i in indices[:disp_count // 2]:
            filename = all_filenames[i]
            score = anomaly_scores[i]
            true_label = y_true[i]
            pred_label = y_pred[i]
            true_status = "異常" if true_label == 1 else "正常"
            pred_status = "異常" if pred_label == 1 else "正常"
            correct = "○" if true_label == pred_label else "×"
            print(f"{filename}: スコア = {score:.4f}, 真値 = {true_status}, 予測 = {pred_status} {correct}")
        
        # 最も異常と判定されたサンプル
        print("\n最も異常と判定されたサンプル:")
        for i in indices[-(disp_count // 2):]:
            filename = all_filenames[i]
            score = anomaly_scores[i]
            true_label = y_true[i]
            pred_label = y_pred[i]
            true_status = "異常" if true_label == 1 else "正常"
            pred_status = "異常" if pred_label == 1 else "正常"
            correct = "○" if true_label == pred_label else "×"
            print(f"{filename}: スコア = {score:.4f}, 真値 = {true_status}, 予測 = {pred_status} {correct}")
    
    def plot_score_distribution(self, anomaly_scores, normal_count, threshold):
        """異常スコアの分布をプロット"""
        plt.figure(figsize=(12, 6))
        
        # 正常と異常でグループ化してプロット
        normal_scores = anomaly_scores[:normal_count]
        abnormal_scores = anomaly_scores[normal_count:]
        
        plt.hist(normal_scores, bins=20, alpha=0.7, label='正常', color='green')
        plt.hist(abnormal_scores, bins=20, alpha=0.7, label='異常', color='red')
        plt.axvline(x=threshold, color='orange', linestyle='--', label=f'閾値 ({threshold:.4f})')
        
        plt.xlabel("異常スコア")
        plt.ylabel("頻度")
        plt.title("異常スコア分布")
        plt.legend()
        plt.tight_layout()
        
        # 出力ディレクトリに保存
        output_path = os.path.join(self.config.output_dir, 'anomaly_score_distribution.png')
        plt.savefig(output_path)
        plt.close()
    
    def plot_feature_space(self, test_features, normal_count):
        """PCA特徴空間の散布図をプロット"""
        if test_features.shape[1] < 2:
            print("警告: 特徴次元が2次元以下のため、特徴空間の可視化をスキップします。")
            return
        
        plt.figure(figsize=(10, 8))
        plt.scatter(test_features[:normal_count, 0], 
                   test_features[:normal_count, 1], 
                   c='green', marker='o', alpha=0.7, label='正常')
        plt.scatter(test_features[normal_count:, 0], 
                   test_features[normal_count:, 1], 
                   c='red', marker='x', alpha=0.7, label='異常')
        plt.title('PCA特徴空間における正常・異常サンプルの分布')
        plt.xlabel('第1主成分')
        plt.ylabel('第2主成分')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 出力ディレクトリに保存
        output_path = os.path.join(self.config.output_dir, 'feature_space_distribution.png')
        plt.savefig(output_path)
        plt.close()


def generate_synthetic_anomalies(normal_features, n_samples=200):
    """正常特徴から合成異常特徴を生成"""
    synthetic_anomalies = []
    
    # 正常特徴量の統計を計算
    mean_vector = np.mean(normal_features, axis=0)
    std_vector = np.std(normal_features, axis=0)
    
    # 1. 境界サンプル生成（正常サンプルを外側に拡張）
    for i in range(n_samples // 3):
        # ランダムに正常サンプルを選択
        idx = np.random.randint(0, len(normal_features))
        normal_sample = normal_features[idx]
        
        # 平均からの方向に拡張
        direction = normal_sample - mean_vector
        # 方向の正規化（ゼロ除算を防ぐ）
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = np.random.randn(*direction.shape)
            direction = direction / np.linalg.norm(direction)
        
        # 拡張係数（1.5〜3.0）
        scale_factor = np.random.uniform(1.5, 3.0)
        boundary_sample = normal_sample + scale_factor * direction * std_vector
        
        synthetic_anomalies.append(boundary_sample)
    
    # 2. クラスター外サンプル生成（遠い異常）
    for i in range(n_samples // 3):
        # 完全なランダムサンプル
        random_sample = mean_vector + np.random.randn(*mean_vector.shape) * 3 * std_vector
        synthetic_anomalies.append(random_sample)
    
    # 3. 局所的に変化させたサンプル
    for i in range(n_samples - len(synthetic_anomalies)):
        # ランダムに正常サンプルを選択
        idx = np.random.randint(0, len(normal_features))
        normal_sample = normal_features[idx].copy()
        
        # 一部のフィーチャーだけを大きく変更
        feature_indices = np.random.choice(
            normal_sample.shape[0], 
            size=max(1, int(normal_sample.shape[0] * 0.1)),  # 10%のフィーチャーを変更
            replace=False
        )
        
        # 選択したフィーチャーの値を5σ変更
        for fidx in feature_indices:
            shift = np.random.choice([-1, 1]) * 5 * std_vector[fidx]
            normal_sample[fidx] += shift
        
        synthetic_anomalies.append(normal_sample)
    
    return np.array(synthetic_anomalies)


def improved_anomaly_detection_pipeline(config):
    """改善された異常検知パイプライン"""
    print("\n====== 改善版異常検知を開始します ======")
    print(str(config))
    
    try:
        # 1. データセット読み込み
        dataset = ImageDataset(config)
        data = dataset.load_dataset()
        print(f"訓練画像数: {len(data['train_imgs'])}")
        print(f"テスト正常画像数: {len(data['test_normal_imgs'])}")
        print(f"テスト異常画像数: {len(data['test_abnormal_imgs'])}")
        
        # 2. 特徴抽出器の初期化
        feature_extractor = FeatureExtractor(config)
        
        # 3. 特徴抽出
        print("正常画像から特徴量を抽出しています...")
        if config.feature_extraction_method == 'autoencoder':
            train_features, train_recon = feature_extractor.extract_features(data['train_imgs'])
        else:
            train_features = feature_extractor.extract_features(data['train_imgs'])
        
        print("テスト画像から特徴量を抽出しています...")
        if config.feature_extraction_method == 'autoencoder':
            test_features, test_recon = feature_extractor.extract_features(data['all_test_imgs'])
        else:
            test_features = feature_extractor.extract_features(data['all_test_imgs'])
        
        # 4. 合成異常サンプル生成（検証用）
        if config.use_synthetic_samples:
            print("合成異常サンプルを生成しています...")
            synthetic_anomalies = generate_synthetic_anomalies(
                train_features, 
                n_samples=min(config.synthetic_sample_count, len(train_features))
            )
            
            # 検証セット作成
            val_normal_indices = np.random.choice(
                len(train_features), 
                size=min(len(train_features) // 5, 100),  # 20%または最大100サンプル
                replace=False
            )
            val_normal = train_features[val_normal_indices]
            val_anomaly = synthetic_anomalies[:len(val_normal)]
            
            val_features = np.vstack([val_normal, val_anomaly])
            val_labels = np.array([0] * len(val_normal) + [1] * len(val_anomaly))
            validation_data = (val_features, val_labels)
        else:
            validation_data = None
        
        # 5. 検出器の初期化と学習
        if config.use_ensemble:
            print("アンサンブル検出器を学習しています...")
            detector = RobustEnsembleDetector(config)
            normal_scores = detector.fit(train_features, validation_data)
        else:
            print("単一検出器を学習しています...")
            detector = AnomalyDetector(config)
            normal_scores = detector.fit(train_features)
            
            # 検出器をオリジナルのパイプラインと合わせるために変換
            # この行は不要になったので削除します - PCAはcompute_distances内で適用されるようになりました
            # if config.use_pca:
            #     test_features = detector.transform(test_features)
        
        # 6. テストデータの異常スコア計算
        print("テストデータの異常スコアを計算しています...")
        anomaly_scores = detector.compute_distances(test_features)
        
        # 7. オートエンコーダを使用する場合は再構成誤差を結合
        if config.feature_extraction_method == 'autoencoder':
            print("再構成誤差と特徴距離を結合しています...")
            # 再構成誤差の計算
            recon_errors = torch.mean(
                torch.abs(data['all_test_imgs'].to(config.device) - test_recon), 
                dim=(1, 2, 3)
            ).cpu().numpy()
            
            # 正規化
            norm_feature_dist = AnomalyDetector.normalize_scores(anomaly_scores)
            norm_recon_errors = AnomalyDetector.normalize_scores(recon_errors)
            
            # 結合（重み付け）
            alpha = 0.7  # 特徴距離の重み
            anomaly_scores = alpha * norm_feature_dist + (1 - alpha) * norm_recon_errors
        
        # 8. スコアの正規化
        anomaly_scores = AnomalyDetector.normalize_scores(anomaly_scores)
        
        # 9. 最適な閾値の計算
        print("最適な閾値を計算しています...")
        if config.threshold is None:
            if validation_data:
                # 検証データを使った閾値最適化
                threshold_selector = OptimizedThresholdSelector(config)
                # 検証データのスコアのみを使用（normal_scores との連結はしない）
                val_scores = AnomalyDetector.normalize_scores(detector.compute_distances(val_features))
                threshold = threshold_selector.find_optimal_threshold(
                    val_labels,
                    val_scores,
                    method=config.threshold_method
                )
            else:
                # 従来の方法
                threshold = AnomalyDetector.compute_adaptive_threshold(
                    normal_scores, method='percentile', percentile=95
                )
        else:
            threshold = config.threshold
        
        print(f"\n閾値: {threshold:.4f}")
        
        # 10. 異常判定（0: 正常, 1: 異常）
        y_pred = np.array([1 if score > threshold else 0 for score in anomaly_scores])
        
        # 11. 評価と可視化
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(data['y_true'], y_pred, anomaly_scores, data['all_filenames'])
        
        # スコア分布のプロット
        evaluator.plot_score_distribution(
            anomaly_scores, len(data['normal_filenames']), threshold
        )
        
        # 特徴空間の可視化（メモリ効率モードでなければ）
        if config.visualization and not config.memory_efficient and train_features.shape[1] >= 2:
            if train_features.shape[1] > 2:
                # t-SNEによる次元削減
                print("t-SNEで特徴空間を可視化しています...")
                try:
                    # データが多すぎる場合はサンプリング
                    max_samples = 1000
                    if len(train_features) + len(test_features) > max_samples:
                        print(f"t-SNE可視化のためにデータをサンプリングします（最大{max_samples}サンプル）")
                        combined_indices = np.random.choice(
                            len(train_features) + len(test_features),
                            max_samples,
                            replace=False
                        )
                        # train_featuresとtest_featuresを結合
                        combined_features = np.vstack([train_features, test_features])
                        # サンプリング
                        sampled_features = combined_features[combined_indices]
                        # ラベルもサンプリング（訓練データは正常）
                        combined_labels = np.concatenate([
                            np.zeros(len(train_features)), 
                            data['y_true']
                        ])
                        sampled_labels = combined_labels[combined_indices]
                        
                        # t-SNE実行
                        tsne = TSNE(n_components=2, random_state=42)
                        tsne_features = tsne.fit_transform(sampled_features)
                        
                        # 訓練データとテストデータを区別
                        is_train = combined_indices < len(train_features)
                        
                        # プロット
                        plt.figure(figsize=(10, 8))
                        
                        # 訓練データ（すべて正常）
                        plt.scatter(
                            tsne_features[is_train, 0], tsne_features[is_train, 1],
                            c='blue', marker='o', alpha=0.5, label='訓練データ（正常）'
                        )
                        
                        # テストデータを正常と異常に分類
                        test_indices = ~is_train
                        test_normal = test_indices & (sampled_labels == 0)
                        test_abnormal = test_indices & (sampled_labels == 1)
                        
                        plt.scatter(
                            tsne_features[test_normal, 0], tsne_features[test_normal, 1],
                            c='green', marker='o', alpha=0.7, label='テスト（正常）'
                        )
                        plt.scatter(
                            tsne_features[test_abnormal, 0], tsne_features[test_abnormal, 1],
                            c='red', marker='x', alpha=0.7, label='テスト（異常）'
                        )
                    else:
                        # サンプリングなしでt-SNE実行
                        tsne = TSNE(n_components=2, random_state=42)
                        combined_features = np.vstack([train_features, test_features])
                        tsne_features = tsne.fit_transform(combined_features)
                        
                        # 訓練データとテストデータを分離
                        train_tsne = tsne_features[:len(train_features)]
                        test_tsne = tsne_features[len(train_features):]
                        
                        # プロット
                        plt.figure(figsize=(10, 8))
                        plt.scatter(
                            train_tsne[:, 0], train_tsne[:, 1],
                            c='blue', marker='o', alpha=0.5, label='訓練データ（正常）'
                        )
                        
                        # テストデータを正常と異常に分類
                        normal_mask = data['y_true'] == 0
                        plt.scatter(
                            test_tsne[normal_mask, 0], test_tsne[normal_mask, 1],
                            c='green', marker='o', alpha=0.7, label='テスト（正常）'
                        )
                        plt.scatter(
                            test_tsne[~normal_mask, 0], test_tsne[~normal_mask, 1],
                            c='red', marker='x', alpha=0.7, label='テスト（異常）'
                        )
                    
                    plt.title('t-SNEによる特徴空間の可視化')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(config.output_dir, 'tsne_visualization.png'))
                    plt.close()
                    
                except Exception as e:
                    print(f"t-SNE可視化エラー: {str(e)}")
            else:
                # 2次元の場合は直接プロット
                evaluator.plot_feature_space(test_features, len(data['normal_filenames']))
        
        # 12. オートエンコーダ再構成結果の可視化（メモリ効率モードでなければ）
        if config.feature_extraction_method == 'autoencoder' and not config.memory_efficient:
            print("再構成結果を可視化しています...")
            try:
                # サンプル数の決定
                n_samples = min(5, len(data['test_normal_imgs']), len(data['test_abnormal_imgs']))
                
                plt.figure(figsize=(15, 3 * n_samples))
                
                # グリッド設定
                for i in range(n_samples):
                    # 正常サンプル
                    # 元画像
                    plt.subplot(n_samples, 6, i*6 + 1)
                    img = data['test_normal_imgs'][i].permute(1, 2, 0).cpu().numpy()
                    img = img * np.array(config.norm_std) + np.array(config.norm_mean)
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    plt.title("正常 - 原画像" if i == 0 else "")
                    plt.axis('off')
                    
                    # 再構成画像
                    plt.subplot(n_samples, 6, i*6 + 2)
                    recon_img = test_recon[i].permute(1, 2, 0).cpu().numpy()
                    plt.imshow(recon_img)
                    plt.title("正常 - 再構成" if i == 0 else "")
                    plt.axis('off')
                    
                    # 差分
                    plt.subplot(n_samples, 6, i*6 + 3)
                    diff = np.abs(img - recon_img)
                    plt.imshow(diff)
                    plt.title("正常 - 差分" if i == 0 else "")
                    plt.axis('off')
                    
                    # 異常サンプル
                    # 元画像
                    plt.subplot(n_samples, 6, i*6 + 4)
                    img = data['test_abnormal_imgs'][i].permute(1, 2, 0).cpu().numpy()
                    img = img * np.array(config.norm_std) + np.array(config.norm_mean)
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    plt.title("異常 - 原画像" if i == 0 else "")
                    plt.axis('off')
                    
                    # 再構成画像
                    plt.subplot(n_samples, 6, i*6 + 5)
                    recon_img = test_recon[len(data['test_normal_imgs'])+i].permute(1, 2, 0).cpu().numpy()
                    plt.imshow(recon_img)
                    plt.title("異常 - 再構成" if i == 0 else "")
                    plt.axis('off')
                    
                    # 差分
                    plt.subplot(n_samples, 6, i*6 + 6)
                    diff = np.abs(img - recon_img)
                    plt.imshow(diff)
                    plt.title("異常 - 差分" if i == 0 else "")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(config.output_dir, 'reconstruction_results.png'))
                plt.close()
                
            except Exception as e:
                print(f"再構成画像の可視化中にエラーが発生しました: {str(e)}")
        
        return {
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'y_true': data['y_true'],
            'y_pred': y_pred,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"エラー: 異常検知処理中に例外が発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def parameter_tuning(config, param_grid):
    """最適なハイパーパラメータを探索"""
    from sklearn.model_selection import ParameterGrid
    
    best_score = 0
    best_params = None
    results = []
    
    # ParameterGridでハイパーパラメータの組み合わせを生成
    for params in ParameterGrid(param_grid):
        print(f"\n---- パラメータ: {params} ----")
        
        # 現在のパラメータで設定を更新
        current_config = copy.deepcopy(config)
        for key, value in params.items():
            setattr(current_config, key, value)
        
        # 異常検知を実行
        result = improved_anomaly_detection_pipeline(current_config)
        
        if result:
            # 評価指標を計算（ここではF1スコアを使用）
            f1_score = result['metrics']['f1']
            print(f"F1スコア: {f1_score:.4f}")
            
            results.append({
                'params': params,
                'f1': f1_score,
                'results': result
            })
            
            # 最良のパラメータを更新
            if f1_score > best_score:
                best_score = f1_score
                best_params = params
    
    print(f"\n====== パラメータチューニング結果 ======")
    print(f"最適パラメータ: {best_params}")
    print(f"最高F1スコア: {best_score:.4f}")
    
    return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='改善版ViT異常検知')
    parser.add_argument('--train', type=str, default='./data/train/normal', help='正常画像のフォルダパス')
    parser.add_argument('--test_normal', type=str, default='./data/test/normal', help='テスト用正常画像のフォルダパス')
    parser.add_argument('--test_abnormal', type=str, default='./data/test/abnormal', help='テスト用異常画像のフォルダパス')
    parser.add_argument('--threshold_method', type=str, default='balanced_accuracy',
                        choices=['percentile', 'zscore', 'iqr', 'gmm', 'f1', 'balanced_accuracy', 'pr_auc'],
                        help='閾値決定方法')
    parser.add_argument('--threshold', type=float, default=None, help='異常判定の閾値（指定なしの場合は自動計算）')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='使用するViTモデル')
    parser.add_argument('--feature_extraction', type=str, default='enhanced',
                        choices=['original', 'enhanced', 'autoencoder'],
                        help='特徴抽出方法')
    parser.add_argument('--use_pca', action='store_true', help='PCAによる次元削減を使用する')
    parser.add_argument('--max_pca_dim', type=int, default=100, help='PCAで削減する最大次元数')
    parser.add_argument('--use_ensemble', action='store_true', help='アンサンブル検出器を使用する')
    parser.add_argument('--memory_efficient', action='store_true', help='メモリ効率を優先（特徴抽出と検出器を簡略化）')
    parser.add_argument('--output_dir', type=str, default='./output', help='結果の出力ディレクトリ')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--parameter_tuning', action='store_true', help='パラメータチューニングを実行する')
    
    args = parser.parse_args()
    
    # 設定オブジェクトの作成
    config = Config()
    config.train_folder = args.train
    config.test_normal_folder = args.test_normal
    config.test_abnormal_folder = args.test_abnormal
    config.threshold_method = args.threshold_method
    config.threshold = args.threshold
    config.model_name = args.model
    config.feature_extraction_method = args.feature_extraction
    config.use_pca = args.use_pca
    config.max_pca_dim = args.max_pca_dim
    config.use_ensemble = args.use_ensemble
    config.memory_efficient = args.memory_efficient
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    
    # 出力ディレクトリの作成
    os.makedirs(config.output_dir, exist_ok=True)
    
    if args.parameter_tuning:
        # パラメータチューニングの実行
        print("\n====== パラメータチューニングを開始します ======")
        param_grid = {
            'feature_extraction_method': ['original', 'enhanced'],
            'use_ensemble': [True, False],
            'threshold_method': ['balanced_accuracy', 'f1'],
            'use_pca': [True],
            'max_pca_dim': [50, 100],
            'memory_efficient': [True]
        }
        results = parameter_tuning(config, param_grid)
    else:
        # 異常検知を実行
        results = improved_anomaly_detection_pipeline(config)
    
    if results:
        print("\n====== 異常検知が完了しました ======")
        print(f"結果のグラフは {config.output_dir} ディレクトリに保存されました。")


if __name__ == "__main__":
    main()