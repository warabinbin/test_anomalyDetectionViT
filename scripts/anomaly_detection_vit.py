#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py

このスクリプトはViTを用いた異常画像検知を行います。
"""


import os
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, 
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
        
        # 異常検知設定
        self.use_pca = True
        self.pca_variance = 0.95
        self.knn_threshold = 200  # データ数がこれ以下の場合はkNNを使用
        
        # データパス設定
        self.train_folder = './data/train/normal'
        self.test_normal_folder = './data/test/normal'
        self.test_abnormal_folder = './data/test/abnormal'
        
        # 閾値設定
        self.threshold = None
        self.threshold_method = 'percentile'
        
        # 出力設定
        self.output_dir = './output'
        os.makedirs(self.output_dir, exist_ok=True)

    def get_transform(self):
        """画像変換パイプラインを返す"""
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
            f"Use PCA: {self.use_pca}\n"
            f"Threshold method: {self.threshold_method}\n"
            f"Batch size: {self.batch_size}"
        )


class ViTFeatureExtractor(nn.Module):
    """Vision Transformerから特徴を抽出するモデル"""
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


class ImageDataset:
    """画像データセットを管理するクラス"""
    def __init__(self, config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.transform = config.get_transform()
    
    def load_images(self, folder):
        """
        フォルダから画像を読み込む
        
        Args:
            folder: 画像フォルダのパス
            
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
        
        for filename in tqdm(files, desc=f"画像読み込み ({folder})"):
            try:
                path = os.path.join(folder, filename)
                img = Image.open(path).convert('RGB')
                
                transformed_img = self.transform(img)
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
        train_imgs, _ = self.load_images(self.config.train_folder)
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
        self.model = ViTFeatureExtractor(model_name=config.model_name).to(config.device)
        self.model.eval()
        
    def extract_features(self, images):
        """
        画像テンソルから特徴量を抽出
        
        Args:
            images: 画像テンソル
            
        Returns:
            特徴量のNumPy配列
        """
        features = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.config.device)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
                
                # メモリ解放
                del batch
                torch.cuda.empty_cache()
        
        return np.vstack(features)


class AnomalyDetector:
    """異常検知を行うクラス"""
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
            self.pca = PCA(n_components=self.config.pca_variance)
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
            except ValueError as e:
                print(f"MinCovDetエラー: {e}")
                print("代替としてkNN法を使用します")
                self.use_knn = True
                self.nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
                self.nn_model.fit(train_features)
        
        # 正常サンプルの距離を計算（閾値決定用）
        return self.compute_distances(train_features)
    
    def compute_distances(self, features):
        """
        特徴量から異常スコア（距離）を計算
        
        Args:
            features: 特徴量
            
        Returns:
            距離のリスト
        """
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
        auc = roc_auc_score(y_true, anomaly_scores)
        
        print(f"精度 (Accuracy): {accuracy:.4f}")
        print(f"適合率 (Precision): {precision:.4f}")
        print(f"再現率 (Recall): {recall:.4f}")
        print(f"F1スコア: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
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
            'auc': auc
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
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_score, title='ROC曲線'):
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
        
        # 出力ディレクトリに保存
        output_path = os.path.join(self.config.output_dir, 'roc_curve.png')
        plt.savefig(output_path)
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true, y_score, title='適合率-再現率曲線'):
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
        
        # 出力ディレクトリに保存
        output_path = os.path.join(self.config.output_dir, 'precision_recall_curve.png')
        plt.savefig(output_path)
        plt.show()
    
    def _print_prediction_results(self, all_filenames, anomaly_scores, y_true, y_pred):
        """個別の予測結果を表示する関数"""
        print("\n予測結果詳細:")
        for filename, score, true_label, pred_label in zip(all_filenames, anomaly_scores, y_true, y_pred):
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
        plt.show()
    
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
        plt.show()


def anomaly_detection_pipeline(config):
    """
    異常検知のパイプライン処理
    
    Args:
        config: 設定オブジェクト
        
    Returns:
        結果の辞書
    """
    print("\n====== 異常検知を開始します ======")
    print(str(config))
    
    try:
        # データセットの読み込み
        dataset = ImageDataset(config)
        data = dataset.load_dataset()
        print(f"訓練画像数: {len(data['train_imgs'])}")
        print(f"テスト正常画像数: {len(data['test_normal_imgs'])}")
        print(f"テスト異常画像数: {len(data['test_abnormal_imgs'])}")
        
        # 特徴抽出
        feature_extractor = FeatureExtractor(config)
        
        print("正常画像から特徴量を抽出しています...")
        train_features = feature_extractor.extract_features(data['train_imgs'])
        
        print("テスト画像から特徴量を抽出しています...")
        test_features = feature_extractor.extract_features(data['all_test_imgs'])
        
        # 異常検知モデルの構築
        detector = AnomalyDetector(config)
        normal_distances = detector.fit(train_features)
        
        # テスト特徴量の変換（PCA使用時）
        test_features = detector.transform(test_features)
        
        # 異常スコア計算
        anomaly_scores = detector.compute_distances(test_features)
        
        # スコアの正規化
        anomaly_scores = detector.normalize_scores(anomaly_scores)
        
        # 閾値決定
        if config.threshold is None:
            # 正常データのスコアも正規化
            normal_scores = detector.normalize_scores(normal_distances)
            # 適応的閾値計算
            threshold = detector.compute_adaptive_threshold(normal_scores, method=config.threshold_method)
        else:
            threshold = config.threshold
        
        print(f"\n閾値: {threshold:.4f}")
        
        # 異常判定（0: 正常, 1: 異常）
        y_pred = np.array([1 if score > threshold else 0 for score in anomaly_scores])
        
        # 結果の評価
        evaluator = Evaluator(config)
        
        # 結果の詳細表示
        evaluator._print_prediction_results(
            data['all_filenames'], anomaly_scores, data['y_true'], y_pred
        )
        
        # スコア分布のプロット
        evaluator.plot_score_distribution(
            anomaly_scores, len(data['normal_filenames']), threshold
        )
        
        # 特徴空間のプロット（PCA使用時）
        if config.use_pca:
            evaluator.plot_feature_space(test_features, len(data['normal_filenames']))
        
        # 性能評価
        print("\n====== 精度評価 ======")
        metrics = evaluator.evaluate(data['y_true'], y_pred, anomaly_scores)
        
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


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ViTを使用した異常検知')
    parser.add_argument('--train', type=str, default='./data/train/normal', help='正常画像のフォルダパス')
    parser.add_argument('--test_normal', type=str, default='./data/test/normal', help='テスト用正常画像のフォルダパス')
    parser.add_argument('--test_abnormal', type=str, default='./data/test/abnormal', help='テスト用異常画像のフォルダパス')
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        choices=['percentile', 'zscore', 'iqr', 'gmm'],
                        help='閾値決定方法')
    parser.add_argument('--threshold', type=float, default=None, help='異常判定の閾値（指定なしの場合は自動計算）')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='使用するViTモデル')
    parser.add_argument('--use_pca', action='store_true', help='PCAによる次元削減を使用する')
    parser.add_argument('--output_dir', type=str, default='./output', help='結果の出力ディレクトリ')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    
    args = parser.parse_args()
    
    # 設定オブジェクトの作成
    config = Config()
    config.train_folder = args.train
    config.test_normal_folder = args.test_normal
    config.test_abnormal_folder = args.test_abnormal
    config.threshold_method = args.threshold_method
    config.threshold = args.threshold
    config.model_name = args.model
    config.use_pca = args.use_pca
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    
    # 出力ディレクトリの作成
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 異常検知を実行
    results = anomaly_detection_pipeline(config)
    
    if results:
        print("\n====== 異常検知が完了しました ======")
        print(f"結果のグラフは {config.output_dir} ディレクトリに保存されました。")


if __name__ == "__main__":
    main()