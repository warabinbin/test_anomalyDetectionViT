#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
webui_data_generator.py

このスクリプトはSeleniumを使用してWebGUIからスクリーンショットを取得し、
データセットを構築します。正常HTMLからは正常データを、異常HTMLからは異常データを取得します。
"""

import os
import time
import random
import logging
import argparse
from io import BytesIO
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementNotInteractableException
)
from webdriver_manager.chrome import ChromeDriverManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("webui_data_generator.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("WebUIDataGenerator")

# 定数
DEFAULT_NORMAL_HTML = "testapp/simple-storage-manager.html"
DEFAULT_ABNORMAL_HTML = "testapp/error-simple-storage-manager.html"
DEFAULT_DATASET_DIR = "data"
DEFAULT_IMG_SIZE = (224, 224)  # ViTのデフォルトサイズ
DEFAULT_NUM_SAMPLES = 500
DEFAULT_TRAIN_RATIO = 0.7  # 訓練データの割合
WAIT_TIMEOUT = 10
INTERACTION_PAUSE = 0.5  # アクション間の待機時間
UI_TEXT_EXAMPLES = [
    "買い物リスト",
    "ToDo: 週末のタスク",
    "会議の議事録",
    "プロジェクト計画",
    "読書メモ",
    "料理レシピ",
    "旅行計画",
    "学習ノート",
    "アイデアメモ",
    "連絡先リスト"
]
SAMPLE_TEXTS = [
    "これはテストメモです。保存して表示できるか確認します。",
    "プロジェクトのタスク:\n1. 要件分析\n2. 設計\n3. 実装\n4. テスト",
    "会議のアジェンダ:\n- 前回の議事録確認\n- 進捗状況の共有\n- 課題の討議",
    "買い物リスト:\n・牛乳\n・卵\n・パン\n・りんご\n・バナナ",
    "今日の目標:\n1. メールチェック\n2. レポート作成\n3. 資料の準備\n4. 会議出席",
    "メモ: 明日の会議は9時からオンラインで開催されます。資料を事前に確認しておくこと。",
    "読書メモ:\nタイトル: データサイエンス入門\n著者: 山田太郎\n重要ポイント: 第3章のアルゴリズム解説が特に参考になる。",
    "旅行計画:\n・出発: 6月10日\n・帰宅: 6月15日\n・宿泊: グランドホテル\n・持ち物: パスポート、カメラ、充電器",
    "学習計画:\n1. 基礎理論の復習\n2. 問題演習\n3. モデル構築\n4. パラメータ調整\n5. 性能評価",
    "ランダムなメモテキストです。アプリケーションのテストに使用します。様々な操作を行います。"
]


class WebUIDataCollector:
    """
    WebGUIからデータを収集するクラス
    
    正常HTMLからは正常データを、異常HTMLからは異常データを取得します。
    """
    
    def __init__(
        self,
        normal_html: str,
        abnormal_html: str,
        dataset_dir: str,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        num_samples: int = DEFAULT_NUM_SAMPLES
    ):
        """
        初期化メソッド
        
        Args:
            normal_html: 正常なUIのHTMLファイルパス
            abnormal_html: 異常なUIのHTMLファイルパス
            dataset_dir: データセットを保存するディレクトリパス
            img_size: 生成する画像サイズ (幅, 高さ)
            train_ratio: 訓練データの割合 (0.0〜1.0)
            num_samples: 生成するサンプル数
        """
        self.normal_html = os.path.abspath(normal_html)
        self.abnormal_html = os.path.abspath(abnormal_html)
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.num_samples = num_samples
        
        # HTMLファイルが存在するか確認
        if not os.path.exists(self.normal_html):
            raise FileNotFoundError(f"正常HTMLファイルが見つかりません: {self.normal_html}")
        if not os.path.exists(self.abnormal_html):
            raise FileNotFoundError(f"異常HTMLファイルが見つかりません: {self.abnormal_html}")
            
        logger.info(f"正常HTMLファイル: {self.normal_html}")
        logger.info(f"異常HTMLファイル: {self.abnormal_html}")
        
        # データセットディレクトリ構造
        self.train_normal_dir = os.path.join(dataset_dir, "train", "normal")
        self.test_normal_dir = os.path.join(dataset_dir, "test", "normal")
        self.test_abnormal_dir = os.path.join(dataset_dir, "test", "abnormal")
        
        # ディレクトリ作成
        os.makedirs(self.train_normal_dir, exist_ok=True)
        os.makedirs(self.test_normal_dir, exist_ok=True)
        os.makedirs(self.test_abnormal_dir, exist_ok=True)
        
        # Webドライバー設定
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # ヘッドレスモードで実行する場合はコメントアウト
        chrome_options.add_argument("--window-size=1366,768")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--lang=ja")  # 日本語設定
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
    
    def load_page(self, html_path: str) -> bool:
        """
        HTMLページをロード
        
        Args:
            html_path: HTMLファイルのパス
            
        Returns:
            bool: ロードに成功した場合はTrue
        """
        try:
            # 絶対パスに変換して確実にfileプロトコルで開く
            abs_path = os.path.abspath(html_path)
            file_url = f"file:///{abs_path.replace(os.sep, '/')}"
            logger.info(f"ページをロードしています: {file_url}")
            self.driver.get(file_url)
            
            # ページが読み込まれるまで待機
            WebDriverWait(self.driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            logger.info("ページのロードに成功しました")
            return True
        except TimeoutException:
            logger.error(f"ページのロードがタイムアウトしました: {html_path}")
            return False
        except Exception as e:
            logger.error(f"ページのロード中にエラーが発生しました: {str(e)}")
            return False
    
    def take_screenshot(self, output_path: str):
        """
        スクリーンショットを撮影して保存
        
        Args:
            output_path: 保存先のパス
        """
        try:
            # スクリーンショットを撮影
            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(BytesIO(screenshot))
            
            # 指定サイズにリサイズ
            image = image.resize(self.img_size, Image.LANCZOS)
            
            # 保存先ディレクトリを確認
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存
            image.save(output_path)
            logger.debug(f"スクリーンショットを保存しました: {output_path}")
            return image
        except Exception as e:
            logger.error(f"スクリーンショットの撮影中にエラーが発生しました: {str(e)}")
            return None
    
    def perform_random_interaction(self) -> str:
        """
        UIでランダムなインタラクションを実行
        
        Returns:
            str: 実行したアクションの説明
        """
        try:
            # 実行可能なアクションのリスト
            actions = [
                self.input_text_to_title,
                self.input_text_to_content,
                self.click_save_button,
                self.search_text,
                self.view_text,
                self.delete_text,
                self.close_modal,
                self.scroll_page
            ]
            
            # ランダムにアクションを選択して実行
            action = random.choice(actions)
            result = action()
            time.sleep(INTERACTION_PAUSE)  # アクション後の短い待機
            return result
        except Exception as e:
            logger.warning(f"インタラクション中にエラーが発生しました: {str(e)}")
            return "インタラクションに失敗"
    
    def input_text_to_title(self) -> str:
        """タイトル入力フィールドにテキストを入力"""
        try:
            title_input = self.driver.find_element(By.ID, "text-title")
            title_input.clear()
            title = random.choice(UI_TEXT_EXAMPLES)
            title_input.send_keys(title)
            return f"タイトルを入力: {title}"
        except NoSuchElementException:
            return "タイトル入力フィールドが見つかりません"
        except ElementNotInteractableException:
            return "タイトル入力フィールドが操作できません"
    
    def input_text_to_content(self) -> str:
        """コンテンツ入力フィールドにテキストを入力"""
        try:
            content_input = self.driver.find_element(By.ID, "text-input")
            content_input.clear()
            content = random.choice(SAMPLE_TEXTS)
            content_input.send_keys(content)
            return "コンテンツを入力"
        except NoSuchElementException:
            return "コンテンツ入力フィールドが見つかりません"
        except ElementNotInteractableException:
            return "コンテンツ入力フィールドが操作できません"
    
    def click_save_button(self) -> str:
        """保存ボタンをクリック"""
        try:
            save_btn = self.driver.find_element(By.ID, "save-btn")
            save_btn.click()
            # アラートが表示された場合は対応
            try:
                alert = WebDriverWait(self.driver, 3).until(EC.alert_is_present())
                alert.accept()
                return "保存ボタンをクリック (アラート表示)"
            except TimeoutException:
                return "保存ボタンをクリック"
        except NoSuchElementException:
            return "保存ボタンが見つかりません"
        except ElementNotInteractableException:
            return "保存ボタンが操作できません"
    
    def search_text(self) -> str:
        """テキストを検索"""
        try:
            search_input = self.driver.find_element(By.ID, "search-input")
            search_input.clear()
            search_term = random.choice([
                "", 
                "リスト", 
                "メモ", 
                "テスト", 
                "プロジェクト"
            ])
            search_input.send_keys(search_term)
            search_btn = self.driver.find_element(By.ID, "search-btn")
            search_btn.click()
            return f"テキストを検索: {search_term}"
        except NoSuchElementException:
            return "検索フィールドが見つかりません"
        except ElementNotInteractableException:
            return "検索フィールドが操作できません"
    
    def view_text(self) -> str:
        """テキストを表示"""
        try:
            view_btns = self.driver.find_elements(By.CLASS_NAME, "view-btn")
            if view_btns:
                view_btn = random.choice(view_btns)
                view_btn.click()
                return "テキストを表示"
            return "表示可能なテキストがありません"
        except NoSuchElementException:
            return "表示ボタンが見つかりません"
        except ElementNotInteractableException:
            return "表示ボタンが操作できません"
    
    def delete_text(self) -> str:
        """テキストを削除"""
        try:
            delete_btns = self.driver.find_elements(By.CLASS_NAME, "delete-btn")
            if delete_btns:
                delete_btn = random.choice(delete_btns)
                delete_btn.click()
                # 確認ダイアログに対応
                try:
                    alert = WebDriverWait(self.driver, 3).until(EC.alert_is_present())
                    alert.accept()
                    return "テキストを削除 (確認済み)"
                except TimeoutException:
                    return "テキストを削除"
            return "削除可能なテキストがありません"
        except NoSuchElementException:
            return "削除ボタンが見つかりません"
        except ElementNotInteractableException:
            return "削除ボタンが操作できません"
    
    def close_modal(self) -> str:
        """モーダルを閉じる"""
        try:
            # モーダルが表示されているか確認
            modal = self.driver.find_element(By.ID, "view-modal")
            if modal.is_displayed():
                close_btn = self.driver.find_element(By.CLASS_NAME, "close-modal")
                close_btn.click()
                return "モーダルを閉じました"
            return "表示されているモーダルがありません"
        except NoSuchElementException:
            return "モーダルまたは閉じるボタンが見つかりません"
        except ElementNotInteractableException:
            return "モーダルまたは閉じるボタンが操作できません"
    
    def scroll_page(self) -> str:
        """ページをスクロール"""
        try:
            scroll_amount = random.randint(-300, 300)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount})")
            return f"ページをスクロール: {scroll_amount}px"
        except Exception as e:
            return f"スクロール中にエラー: {str(e)}"
    
    def collect_data(self):
        """
        データセットを生成
        """
        try:
            # 正常UIからのデータ収集
            logger.info("正常UIからデータを収集しています...")
            self._collect_normal_data()
            
            # 異常UIからのデータ収集
            logger.info("異常UIからデータを収集しています...")
            self._collect_abnormal_data()
            
            # 結果の要約を表示
            train_normal_count = len(os.listdir(self.train_normal_dir))
            test_normal_count = len(os.listdir(self.test_normal_dir))
            test_abnormal_count = len(os.listdir(self.test_abnormal_dir))
            
            print("\nデータセット情報:")
            print(f"  - 訓練データ (正常): {train_normal_count}枚")
            print(f"  - テストデータ (正常): {test_normal_count}枚")
            print(f"  - テストデータ (異常): {test_abnormal_count}枚")
            print(f"  - データセット総数: {train_normal_count + test_normal_count + test_abnormal_count}枚")
            print(f"  - 出力先: {os.path.abspath(self.dataset_dir)}")
            
        except Exception as e:
            logger.error(f"データ収集中にエラーが発生しました: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ブラウザを閉じる
            self.driver.quit()
    
    def _collect_normal_data(self):
        """正常UIからのサンプルを収集"""
        if not self.load_page(self.normal_html):
            logger.error("正常UIのロードに失敗しました。")
            return
        
        # 訓練データと検証データに分割
        num_train = int(self.num_samples * self.train_ratio)
        num_test = self.num_samples - num_train
        
        # 訓練サンプルを生成
        logger.info(f"正常UI: 訓練サンプルを {num_train} 個生成しています...")
        for i in tqdm(range(num_train)):
            # ランダムなUIインタラクションを実行
            for _ in range(random.randint(3, 10)):
                self.perform_random_interaction()
            
            # スクリーンショットを撮影
            filename = f"normal_train_{i:04d}.png"
            output_path = os.path.join(self.train_normal_dir, filename)
            self.take_screenshot(output_path)
        
        # テストサンプル（正常）を生成
        logger.info(f"正常UI: テストサンプルを {num_test} 個生成しています...")
        for i in tqdm(range(num_test)):
            # ランダムなUIインタラクションを実行
            for _ in range(random.randint(3, 10)):
                self.perform_random_interaction()
            
            # スクリーンショットを撮影
            filename = f"normal_test_{i:04d}.png"
            output_path = os.path.join(self.test_normal_dir, filename)
            self.take_screenshot(output_path)
    
    def _collect_abnormal_data(self):
        """異常UIからのサンプルを収集"""
        if not self.load_page(self.abnormal_html):
            logger.error("異常UIのロードに失敗しました。")
            return
        
        # 異常データはテスト用にのみ収集
        logger.info(f"異常UI: テストサンプルを {self.num_samples} 個生成しています...")
        for i in tqdm(range(self.num_samples)):
            # ランダムなUIインタラクションを実行
            for _ in range(random.randint(3, 10)):
                self.perform_random_interaction()
            
            # スクリーンショットを撮影
            filename = f"abnormal_test_{i:04d}.png"
            output_path = os.path.join(self.test_abnormal_dir, filename)
            self.take_screenshot(output_path)
    
    def apply_data_augmentation(self):
        """
        訓練データに対してデータ拡張を適用
        """
        logger.info("訓練データの拡張を開始しています...")
        
        # 既存の訓練サンプルを取得
        original_files = [f for f in os.listdir(self.train_normal_dir) if f.endswith('.png')]
        
        if not original_files:
            logger.error("拡張する訓練データが見つかりません。")
            return
        
        logger.info(f"{len(original_files)}枚の訓練データに対してデータ拡張を実行しています...")
        
        # 各画像に対してデータ拡張を実行
        for filename in tqdm(original_files):
            source_path = os.path.join(self.train_normal_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                # 画像を読み込み
                image = Image.open(source_path)
                
                # 1. 回転（軽度な角度）
                for angle in [-5, 5, -10, 10]:
                    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
                    aug_filename = f"{base_name}_rot{angle}.png"
                    rotated.save(os.path.join(self.train_normal_dir, aug_filename))
                
                # 2. 明るさ調整
                for factor in [0.8, 1.2]:
                    # PILで明るさ調整の実装
                    brightness_adj = Image.eval(image, lambda x: min(255, max(0, int(x * factor))))
                    aug_filename = f"{base_name}_bright{int(factor*100)}.png"
                    brightness_adj.save(os.path.join(self.train_normal_dir, aug_filename))
                
                # 3. コントラスト調整
                for factor in [0.8, 1.2]:
                    # PILでのコントラスト調整の簡易実装
                    gray_avg = int(np.array(image).mean())
                    contrast_adj = Image.eval(image, lambda x: min(255, max(0, gray_avg + (x - gray_avg) * factor)))
                    aug_filename = f"{base_name}_contrast{int(factor*100)}.png"
                    contrast_adj.save(os.path.join(self.train_normal_dir, aug_filename))
                
                # 4. ランダムクロップ
                width, height = image.size
                crop_size = (int(width * 0.9), int(height * 0.9))
                for i in range(2):
                    left = random.randint(0, width - crop_size[0])
                    top = random.randint(0, height - crop_size[1])
                    cropped = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
                    cropped = cropped.resize(self.img_size, Image.LANCZOS)
                    aug_filename = f"{base_name}_crop{i}.png"
                    cropped.save(os.path.join(self.train_normal_dir, aug_filename))
                
            except Exception as e:
                logger.error(f"画像 {filename} の拡張中にエラーが発生しました: {str(e)}")
        
        augmented_count = len(os.listdir(self.train_normal_dir))
        original_count = len(original_files)
        logger.info(f"データ拡張が完了しました。元の画像: {original_count}枚、拡張後: {augmented_count}枚")
    
    def run(self):
        """
        データ収集とデータ拡張を実行
        """
        # データ収集
        self.collect_data()
        
        # データ拡張
        self.apply_data_augmentation()


def list_html_files(directory):
    """指定されたディレクトリ内のHTMLファイルを一覧表示"""
    html_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    
    if html_files:
        print(f"\n指定されたディレクトリ({directory})内のHTMLファイル:")
        for i, file in enumerate(html_files, 1):
            print(f"{i}. {file}")
    else:
        print(f"\n指定されたディレクトリ({directory})内にHTMLファイルが見つかりませんでした。")


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="WebGUIからデータを収集し、データセットを構築するツール"
    )
    parser.add_argument(
        "--normal", 
        type=str, 
        default=DEFAULT_NORMAL_HTML,
        help=f"正常なUIのHTMLファイル (デフォルト: {DEFAULT_NORMAL_HTML})"
    )
    parser.add_argument(
        "--abnormal", 
        type=str, 
        default=DEFAULT_ABNORMAL_HTML,
        help=f"異常なUIのHTMLファイル (デフォルト: {DEFAULT_ABNORMAL_HTML})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_DATASET_DIR,
        help=f"データセット出力ディレクトリ (デフォルト: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=DEFAULT_NUM_SAMPLES,
        help=f"生成するサンプル数 (デフォルト: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=DEFAULT_TRAIN_RATIO,
        help=f"訓練データの割合 (0.0〜1.0) (デフォルト: {DEFAULT_TRAIN_RATIO})"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        nargs=2, 
        default=DEFAULT_IMG_SIZE,
        help=f"画像サイズ (幅 高さ) (デフォルト: {DEFAULT_IMG_SIZE[0]} {DEFAULT_IMG_SIZE[1]})"
    )
    parser.add_argument(
        "--augment-only",
        action="store_true",
        help="データ収集をスキップし、既存データに対してデータ拡張のみを実行"
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="指定されたディレクトリ内のHTMLファイルを表示"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="HTMLファイルを検索するディレクトリ (--list-filesオプションと共に使用)"
    )
    return parser.parse_args()


def main():
    """メイン関数"""
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # HTMLファイルのリストを表示するオプション
    if args.list_files:
        list_html_files(args.dir)
        return
    
    try:
        # データコレクターを初期化
        collector = WebUIDataCollector(
            normal_html=args.normal,
            abnormal_html=args.abnormal,
            dataset_dir=args.output,
            img_size=tuple(args.img_size),
            train_ratio=args.train_ratio,
            num_samples=args.samples
        )
        
        # データ拡張のみモード
        if args.augment_only:
            collector.apply_data_augmentation()
        else:
            # データ収集と拡張を実行
            collector.run()
            
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("\nHTMLファイルパスを確認するために --list-files オプションを使用してください:")
        print("  python webui_data_generator.py --list-files --dir ディレクトリパス")
    
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()