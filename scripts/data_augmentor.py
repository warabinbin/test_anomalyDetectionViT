#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py

このスクリプトは既存の画像データセットに対してデータ拡張処理を適用します。
回転、明るさ調整、コントラスト調整、ランダムクロップなどの拡張を行います。
"""

import os
import random
import logging
import argparse
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_augmentor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("DataAugmentor")

# 定数
DEFAULT_DATASET_DIR = "data"
DEFAULT_IMG_SIZE = (224, 224)  # ViTのデフォルトサイズ

class DataAugmentor:
    """
    画像データセットに対してデータ拡張を適用するクラス
    """
    
    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        augment_test: bool = False
    ):
        """
        初期化メソッド
        
        Args:
            dataset_dir: データセットのディレクトリパス
            img_size: 生成する画像サイズ (幅, 高さ)
            augment_test: テストデータも拡張するかのフラグ
        """
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.augment_test = augment_test
        
        # データセットディレクトリ構造
        self.train_normal_dir = os.path.join(dataset_dir, "train", "normal")
        self.test_normal_dir = os.path.join(dataset_dir, "test", "normal")
        
        # ディレクトリが存在するか確認
        if not os.path.exists(self.train_normal_dir):
            raise FileNotFoundError(f"訓練データディレクトリが見つかりません: {self.train_normal_dir}")
            
        if augment_test and not os.path.exists(self.test_normal_dir):
            raise FileNotFoundError(f"テストデータディレクトリが見つかりません: {self.test_normal_dir}")
            
        logger.info(f"データセットディレクトリ: {self.dataset_dir}")
        logger.info(f"テストデータも拡張: {self.augment_test}")
    
    def apply_data_augmentation(self, target_dir: str):
        """
        指定されたディレクトリ内の画像に対してデータ拡張を適用
        
        Args:
            target_dir: 拡張対象の画像が格納されているディレクトリパス
        """
        logger.info(f"ディレクトリ {target_dir} のデータ拡張を開始...")
        
        # 既存の画像ファイルを取得
        original_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not original_files:
            logger.error(f"拡張する画像データが見つかりません: {target_dir}")
            return
        
        logger.info(f"{len(original_files)}枚の画像に対してデータ拡張を実行しています...")
        
        # 各画像に対してデータ拡張を実行
        for filename in tqdm(original_files):
            source_path = os.path.join(target_dir, filename)
            base_name = os.path.splitext(filename)[0]
            file_ext = os.path.splitext(filename)[1]
            
            try:
                # 画像を読み込み
                image = Image.open(source_path)
                
                # 回転（軽度な角度）
                self._apply_rotation(image, base_name, file_ext, target_dir)
                
                # 明るさ調整
                self._apply_brightness(image, base_name, file_ext, target_dir)
                
                # コントラスト調整
                self._apply_contrast(image, base_name, file_ext, target_dir)
                
                # ランダムクロップ
                self._apply_random_crop(image, base_name, file_ext, target_dir)
                
            except Exception as e:
                logger.error(f"画像 {filename} の拡張中にエラーが発生しました: {str(e)}")
    
    def _apply_rotation(self, image: Image.Image, base_name: str, file_ext: str, target_dir: str):
        """回転を適用"""
        for angle in [-5, 5, -10, 10]:
            try:
                rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
                aug_filename = f"{base_name}_rot{angle}{file_ext}"
                rotated.save(os.path.join(target_dir, aug_filename))
            except Exception as e:
                logger.warning(f"回転適用中にエラー: {str(e)}")
    
    def _apply_brightness(self, image: Image.Image, base_name: str, file_ext: str, target_dir: str):
        """明るさ調整を適用"""
        for factor in [0.8, 1.2]:
            try:
                # PILで明るさ調整の実装
                brightness_adj = Image.eval(image, lambda x: min(255, max(0, int(x * factor))))
                aug_filename = f"{base_name}_bright{int(factor*100)}{file_ext}"
                brightness_adj.save(os.path.join(target_dir, aug_filename))
            except Exception as e:
                logger.warning(f"明るさ調整中にエラー: {str(e)}")
    
    def _apply_contrast(self, image: Image.Image, base_name: str, file_ext: str, target_dir: str):
        """コントラスト調整を適用"""
        for factor in [0.8, 1.2]:
            try:
                # PILでのコントラスト調整の簡易実装
                gray_avg = int(np.array(image).mean())
                contrast_adj = Image.eval(image, lambda x: min(255, max(0, gray_avg + (x - gray_avg) * factor)))
                aug_filename = f"{base_name}_contrast{int(factor*100)}{file_ext}"
                contrast_adj.save(os.path.join(target_dir, aug_filename))
            except Exception as e:
                logger.warning(f"コントラスト調整中にエラー: {str(e)}")
    
    def _apply_random_crop(self, image: Image.Image, base_name: str, file_ext: str, target_dir: str):
        """ランダムクロップを適用"""
        width, height = image.size
        crop_size = (int(width * 0.9), int(height * 0.9))
        
        for i in range(2):
            try:
                left = random.randint(0, width - crop_size[0])
                top = random.randint(0, height - crop_size[1])
                cropped = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
                cropped = cropped.resize(self.img_size, Image.LANCZOS)
                aug_filename = f"{base_name}_crop{i}{file_ext}"
                cropped.save(os.path.join(target_dir, aug_filename))
            except Exception as e:
                logger.warning(f"ランダムクロップ中にエラー: {str(e)}")
    
    def apply_augmentation_to_dataset(self):
        """
        データセット全体に対してデータ拡張を適用
        """
        try:
            # 訓練データに対してデータ拡張を適用
            logger.info("訓練データに対してデータ拡張を適用...")
            self.apply_data_augmentation(self.train_normal_dir)
            
            # テストデータも拡張する場合
            if self.augment_test:
                logger.info("テストデータに対してデータ拡張を適用...")
                self.apply_data_augmentation(self.test_normal_dir)
            
            # 結果の要約を表示
            train_count = len([f for f in os.listdir(self.train_normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            test_count = 0
            if self.augment_test:
                test_count = len([f for f in os.listdir(self.test_normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            print("\nデータ拡張結果:")
            print(f"  - 訓練データ (拡張後): {train_count}枚")
            if self.augment_test:
                print(f"  - テストデータ (拡張後): {test_count}枚")
            print(f"  - 合計: {train_count + test_count}枚")
            print(f"  - データセットディレクトリ: {os.path.abspath(self.dataset_dir)}")
            
        except Exception as e:
            logger.error(f"データ拡張中にエラーが発生しました: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="既存の画像データセットに対してデータ拡張を適用するツール"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=DEFAULT_DATASET_DIR,
        help=f"データセットディレクトリ (デフォルト: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        nargs=2, 
        default=DEFAULT_IMG_SIZE,
        help=f"画像サイズ (幅 高さ) (デフォルト: {DEFAULT_IMG_SIZE[0]} {DEFAULT_IMG_SIZE[1]})"
    )
    parser.add_argument(
        "--augment-test",
        action="store_true",
        help="テストデータも拡張する"
    )
    parser.add_argument(
        "--custom-dir",
        type=str,
        help="特定のディレクトリのみに対してデータ拡張を適用"
    )
    return parser.parse_args()

def main():
    """メイン関数"""
    # コマンドライン引数を解析
    args = parse_arguments()
    
    try:
        # カスタムディレクトリが指定されている場合
        if args.custom_dir:
            if not os.path.exists(args.custom_dir):
                raise FileNotFoundError(f"指定されたディレクトリが見つかりません: {args.custom_dir}")
            
            augmentor = DataAugmentor(
                dataset_dir=os.path.dirname(args.custom_dir),
                img_size=tuple(args.img_size)
            )
            augmentor.apply_data_augmentation(args.custom_dir)
            
        # データセット全体に対して拡張を適用
        else:
            augmentor = DataAugmentor(
                dataset_dir=args.dataset,
                img_size=tuple(args.img_size),
                augment_test=args.augment_test
            )
            augmentor.apply_augmentation_to_dataset()
    
    except FileNotFoundError as e:
        print(f"エラー: {e}")
    
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()