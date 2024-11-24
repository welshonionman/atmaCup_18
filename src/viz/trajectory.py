import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.constants import IMAGE_DIR, TARGET_COLUMNS


def get_images(df_feature_train: pd.DataFrame, idx: int) -> list[Image.Image]:
    """データフレームの指定された行に対応する3つの時系列画像を読み込む

    Args:
        df_feature_train (pd.DataFrame): 特徴量データフレーム。'ID'列が必要。
        idx (int): 読み込みたい画像に対応する行のインデックス

    Returns:
        list[Image.Image]: 3つの時系列画像のリスト。以下の順序で格納:
            - 現在時刻tの画像 (image_t.png)
            - 0.5秒前の画像 (image_t-0.5.png)
            - 1.0秒前の画像 (image_t-1.0.png)

    Note:
        - 画像は'{IMAGE_DIR}/{ID}/'ディレクトリから読み込まれる
        - 各画像のファイル名は'image_t.png'、'image_t-0.5.png'、'image_t-1.0.png'
        - 画像ファイルが存在しない場合、FileNotFoundErrorが発生する
    """
    row = df_feature_train.iloc[idx]
    id_ = row["ID"]
    return [
        Image.open(f"{IMAGE_DIR}/{id_}/image_t-1.0.png"),
        Image.open(f"{IMAGE_DIR}/{id_}/image_t-0.5.png"),
        Image.open(f"{IMAGE_DIR}/{id_}/image_t.png"),
    ]


def camera_to_image(P_camera: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """カメラ座標系の3次元点を画像座標系の2次元点に変換する

    カメラ座標系の3次元点(x,y,z)を、カメラの内部パラメータ行列を用いて画像座標系の2次元点(u,v)に投影変換します。
    この変換は以下のステップで行われます:
    1. 3次元点に内部パラメータ行列を掛けて同次座標系での画像座標を得る
    2. 同次座標の正規化(z座標による除算)を行い2次元画像座標を得る

    同次座標系について:
    - n次元の点を(n+1)次元で表現する座標系
    - 例: 2次元点(x,y)は同次座標では(x,y,1)や(2x,2y,2)など、最後の成分で他を割った値が等しい座標で表現される
    - 射影変換を行列計算で表現できる利点がある
    - 無限遠点も扱える

    Args:
        P_camera (np.ndarray): カメラ座標系での3次元点 (x, y, z)。形状は(3,)のnumpy配列。
        intrinsic_matrix (np.ndarray): カメラの内部パラメータ行列。形状は(3,3)のnumpy配列。
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]
            fx, fy: 焦点距離
            cx, cy: 画像中心(主点)の座標

    Returns:
        np.ndarray: 画像座標系での2次元点 (u, v)。形状は(2,)のnumpy配列。
            u: 画像の横方向の座標(ピクセル)
            v: 画像の縦方向の座標(ピクセル)

    Note:
        - 入力の3次元点はカメラ座標系で表現されている必要があります
        - カメラ座標系のz座標が0以下の点は適切に投影できない可能性があります
        - 同次座標変換により、3次元から2次元への射影を行列演算で表現できます
    """
    P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)
    # 同次座標から2次元画像座標への変換(z成分による正規化)
    P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
    return P_image


def project_trajectory_to_image_coordinate_system(
    trajectory: np.ndarray, intrinsic_matrix: np.ndarray
):
    """車両中心座標系で表現されたtrajectoryをカメラ座標系に投影し、画像座標系に変換する

    Args:
        trajectory (np.ndarray): 車両中心座標系での軌跡データ。形状は(N, 3)のnumpy配列。
            N: 軌跡の点の数
            各行は(x, y, z)の3次元座標を表す
            - x: 車両前方向の座標(m)
            - y: 車両左方向の座標(m)
            - z: 車両上方向の座標(m)
        intrinsic_matrix (np.ndarray): カメラの内部パラメータ行列。形状は(3,3)のnumpy配列。
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]
            fx, fy: 焦点距離
            cx, cy: 画像中心(主点)の座標

    Returns:
        np.ndarray: 画像座標系に投影された軌跡。形状は(M, 2)のnumpy配列。
            M: カメラから見える点の数(M ≤ N)
            各行は(u, v)の2次元画像座標を表す
            - u: 画像の横方向の座標(ピクセル)
            - v: 画像の縦方向の座標(ピクセル)
            ※ カメラから見える点が1つもない場合(全ての点でz≤0)は、空の配列array([], dtype=float64)を返す

    Note:
        - カメラは地上1.22mの高さに設置されているため、z座標をオフセットする
        - 車両中心座標系からカメラ座標系への変換は以下の回転行列で行う:
          [[0, 0, 1],  # カメラのx軸は車両のz軸
           [-1, 0, 0], # カメラのy軸は車両の-x軸
           [0, 1, 0]]  # カメラのz軸は車両のy軸
        - カメラの視野外(z≤0)の点は除外される
    """
    # カメラの設置されている高さ(1.22m)まで座標系をズラす
    trajectory_with_offset = trajectory.copy()
    trajectory_with_offset[:, 2] = trajectory_with_offset[:, 2] + 1.22

    # 座標の取り方を変更する
    road_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    trajectory_camera = trajectory_with_offset @ road_to_camera
    trajectory_image = np.array(
        [camera_to_image(p, intrinsic_matrix) for p in trajectory_camera if p[2] > 0]
    )
    # trajectoryと同じサイズの配列を返す
    if trajectory_image.size == 0:
        return np.zeros((trajectory.shape[0], 2)).astype(np.int32)

    return trajectory_image.astype(np.float32)


def overlay_trajectory(
    trajectory_image: np.ndarray,
    images: list[Image.Image],
    id_: str,
    figsize=(20, 4),
):
    """車両の軌跡を画像上に重ねて表示する

    Args:
        trajectory (np.ndarray): 車両中心座標系での軌跡データ。形状は(N, 3)のnumpy配列。
            N: 軌跡の点の数
            各行は(x, y, z)の3次元座標を表す
        image (Image.Image): 背景となる画像
        intrinsic_matrix (np.ndarray): カメラの内部パラメータ行列。形状は(3,3)のnumpy配列。
        figsize (tuple, optional): 出力図のサイズ(インチ)。デフォルトは(5.12, 2.56)。

    Note:
        - 軌跡は緑色(forestgreen)の点と線で表示される
        - 画像の表示範囲は横0-128ピクセル、縦0-64ピクセル
        - 縦軸は上下が反転して表示される(画像座標系に合わせるため)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05)
    # fig.tight_layout()
    labels = ["t-1.0", "t-0.5", "t"]
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        ax.imshow(images[i])
        ax.set_xlim(0, 128)
        ax.set_ylim(64, 0)
        ax.set_title(labels[i])
        if i == 2:
            ax.plot(
                trajectory_image[:, 0],
                trajectory_image[:, 1],
                marker="o",
                color="forestgreen",
                alpha=1.0,
                markersize=3,
                linestyle="solid",
            )

    plt.suptitle(f"{id_}")
    plt.show()


def get_trajectory(df_feature_train: pd.DataFrame, idx: int) -> np.ndarray:
    """データフレームから指定されたインデックスの軌跡データを取得する

    Args:
        df_feature_train (pd.DataFrame): 学習用特徴量データフレーム。
            TARGET_COLUMNSで指定された列に軌跡の座標情報が含まれている必要がある。
            座標情報は'x_0'、'y_0'、'z_0'のような形式で格納されている。
        idx (int): 取得したい軌跡データのインデックス

    Returns:
        np.ndarray: 軌跡データ。形状は(6, 3)のnumpy配列。
            - 6行は軌跡の6点を表す。各点は0.5秒間隔で、t+0.5秒からt+3.0秒までの位置を表す
            - 3列はそれぞれx, y, z座標を表す
                - x: 自車の進行方向を正とする座標 (単位: メートル)
                - y: 自車の左側方向を正とする座標 (単位: メートル)
                - z: 上向きを正とする座標 (単位: メートル)
            - 行は0から5までの順序でソートされている
            - データ型はfloat32
            - 例:
                [[x_0, y_0, z_0],  # t+0.5秒後の位置
                 [x_1, y_1, z_1],  # t+1.0秒後の位置
                 [x_2, y_2, z_2],  # t+1.5秒後の位置
                 [x_3, y_3, z_3],  # t+2.0秒後の位置
                 [x_4, y_4, z_4],  # t+2.5秒後の位置
                 [x_5, y_5, z_5]]  # t+3.0秒後の位置

    Note:
        TARGET_COLUMNSには以下のような列名が含まれていることを想定:
        ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', ..., 'x_5', 'y_5', 'z_5']
    """
    row = df_feature_train.iloc[idx]

    pivot_df = row[TARGET_COLUMNS].to_frame().reset_index()
    pivot_df.columns = ["coordinate", "value"]
    # 座標軸(x,y,z)と番号(0-5)を正規表現で抽出
    # 例：'x_0' -> axis='x', number='0'
    pivot_df[["axis", "number"]] = pivot_df["coordinate"].str.extract(r"([xyz])_(\d+)")

    # ピボットテーブルを作成：
    # - インデックス：番号(0-5)
    # - カラム：座標軸(x,y,z)
    # - 値：対応する座標値
    trajectory = pivot_df.pivot(index="number", columns="axis", values="value")
    trajectory.index = trajectory.index.astype(int)
    trajectory = trajectory.sort_index().values
    return trajectory.astype(np.float32)
