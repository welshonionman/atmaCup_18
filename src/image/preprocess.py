import json
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from src.constants import LIGHT_DIR, TRAFFIC_CLASS


def get_bbox_points_and_color(
    traffic_light: dict, class_to_idx: dict
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """バウンディングボックスの座標と描画色を計算する関数

    Args:
        traffic_light (dict): 信号機の情報を含む辞書
        class_to_idx (dict): 信号機クラスとインデックスの対応辞書

    Returns:
        tuple: (point1, point2, color)
            - point1 (tuple[int, int]): バウンディングボックスの左上座標
            - point2 (tuple[int, int]): バウンディングボックスの右下座標
            - color (int): 描画色(0-255)
    """
    bbox = traffic_light["bbox"]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    point1 = (x1, y1)
    point2 = (x2, y2)

    idx = class_to_idx[traffic_light["class"]]
    color = 255 - int(255 * (idx / len(TRAFFIC_CLASS)))

    return point1, point2, color


def draw_traffic_light(image: np.ndarray, id_: str) -> np.ndarray:
    """画像に信号機のバウンディングボックスを描画する関数

    指定された画像に、JSONファイルから読み込んだ信号機の位置情報に基づいて
    バウンディングボックスを描画します。信号機の種類に応じて異なる色で描画されます。

    Args:
        image (np.ndarray): 描画対象の画像配列
        id_ (str): 信号機情報が格納されているJSONファイルのID

    処理の流れ:
        1. 指定されたIDのJSONファイルから信号機情報を読み込み
        2. 信号機クラスとインデックスの対応辞書を作成
        3. 各信号機に対して:
            - バウンディングボックスの座標を取得
            - 信号機クラスに応じた描画色を計算
            - 画像上にバウンディングボックスを描画

    Returns:
        np.ndarray: バウンディングボックスが描画された画像配列

    Note:
        - バウンディングボックスは1ピクセルの太さで描画されます
        - 描画色は信号機クラスのインデックスに基づいて0-255の範囲で計算されます
        - 元の画像は上書きされます
    """
    path = f"{LIGHT_DIR}/{id_}.json"
    traffic_lights = json.load(open(path))

    class_to_idx = {cls: idx for idx, cls in enumerate(TRAFFIC_CLASS)}

    for traffic_light in traffic_lights:
        point1, point2, color = get_bbox_points_and_color(traffic_light, class_to_idx)

        cv2.rectangle(image, point1, point2, color=color, thickness=1)

    return image


def read_image_for_cache(path: str) -> tuple[str, np.ndarray]:
    """画像を読み込んでキャッシュ用のタプルを返す関数

    指定されたパスから画像を読み込み、パスと画像データのタプルを返します。
    グレースケールフラグに応じて、グレースケールまたはRGB画像として読み込みます。

    Args:
        path (str): 読み込む画像のファイルパス

    Returns:
        tuple[str, np.ndarray]: (ファイルパス, 画像データ)のタプル
            - パスは入力されたpathをそのまま返します
            - 画像データはグレースケールまたはRGB形式のnumpy配列

    Note:
        - CFG.use_gray_scaleがTrueの場合はグレースケールで読み込み
        - Falseの場合はBGRで読み込んでRGBに変換します
    """

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return (path, image)


def make_video_cache(paths: list[str]) -> dict[str, np.ndarray]:
    """複数の画像を並列読み込みしてキャッシュを作成する関数

    指定された画像パスのリストから、マルチプロセスを使用して
    効率的に画像を読み込み、パスをキーとする辞書を作成します。

    Args:
        paths (list[str]): 読み込む画像ファイルパスのリスト

    処理の流れ:
        1. デバッグ用の色情報を生成して表示
        2. 使用可能なCPUコア数を取得
        3. マルチプロセスプールを作成
        4. 画像を並列読み込み
        5. 進捗状況を表示しながら読み込み結果をリスト化
        6. パスをキーとする辞書に変換

    Returns:
        dict[str, np.ndarray]: {画像パス: 画像データ}の辞書

    Note:
        - マルチプロセスを使用して読み込みを高速化
        - tqdmで進捗状況を表示
        - 画像の読み込みはread_image_for_cache()を使用
    """

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        res = pool.imap_unordered(read_image_for_cache, paths)
        res = tqdm(res)
        res = list(res)

    return dict(res)
