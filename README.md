# atmaCup_18 26th solution

## Environment
- データセットを`/kaggle/input/atmaCup#18_dataset`に配置
- `docker compose up -d`
- コンテナ内で`uv sync`



## 解法
- baseline
  - [CNN only](https://www.guruguru.science/competitions/25/discussions/30a373f3-3bde-4956-a636-1b8f0934750b/)
  - [LightGBM + CNN](https://www.guruguru.science/competitions/25/discussions/03a365c7-27ce-490e-ab6f-e7788ce470c8/)
- CNNモデル
  - エンコーダとしてEfficientNet_B7を使用
    - maxvitを使用するとCNN単体のスコアは出るが、CNN+LGBMのスコアはあまり出なかった
    - [src/image/exp/efficientnet_b7.py](./src/image/exp/efficientnet_b7.py)
- LightGBM
  - 速度・加速度・ハンドル角度から予測した、車両のx座標・y座標・速度の予測値を特徴量として使用
  - [src/table/feature/naive.py](./src/table/feature/naive.py)



## 試したが効かなかったこと

- huggingfaceから入手した深度推定モデルやセマセグモデルでの推定結果を、LGBMの特徴量として使用
  - [src/feature_extract/depth.py](./src/feature_extract/depth.py)
  - [src/feature_extract/semantic.py](./src/feature_extract/semantic.py)

## Training
- [src/image/exp/efficientnet_b7.py](./src/image/exp/efficientnet_b7.py)
- [src/table/exp/best.py](./src/table/exp/best.py)

## 反省点
- エンコーダを変えてCNN単体での精度向上を目指す実験に時間をかけすぎた
- テーブル特徴量の検討にあまり時間を掛けなかった
- 最後に時間がなくoptunaでのハイパーパラメータチューニングが出来なかった



