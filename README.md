# ViTを用いた異常検知テスト
このリポジトリは、Vision Transformer（ViT）を活用した画像の異常検知手法のテスト実装を目的としています。ViTの特徴抽出能力を利用し、画像の異常箇所を検出する実験的なコードが含まれています。

📁 ディレクトリ構成

└── test_anomalyDetectionViT</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── anomaly_detection_vit.py # ViTを用いた異常検知のメインスクリプト</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── webui_data_generator.py # Web UI用のデータ生成スクリプト</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── data/ # webui_data_generator.pyによって生成されたデータセット</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── testapp/ # テスト用アプリケーション関連ファイル</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── .gitignore</br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── README.md</br>
