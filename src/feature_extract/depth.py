from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DPTForDepthEstimation, DPTImageProcessor

from src.constants import DEVICE, IMAGE_DIR, TEST__FEATURES, TRAIN_FEATURES
from src.feature_extract.common import SemanticSegmentationDataset, add_image_paths


def generate_semantic_maps(
    batch: list[tuple[str, dict[str, Image.Image]]],
    model: DPTForDepthEstimation,
    processor: DPTImageProcessor,
    save_dir: str,
):
    """深度推定モデルを使用して画像バッチの深度マップを生成し保存する関数

    バッチ内の各画像に対して深度推定を行い、結果を画像として保存します。
    各画像は3つの時点(t, t-0.5, t-1.0)について処理されます。

    Args:
        batch (list[tuple[str, dict[str, Image.Image]]]): 処理する画像バッチ。
            各要素は(ID, images)のタプルで、IDは画像の識別子、imagesは時点ごとの画像を含む辞書
        model (DPTForDepthEstimation): 深度推定モデル
        processor (DPTImageProcessor): 画像の前処理を行うプロセッサ
        save_dir (str): 生成された深度マップを保存するディレクトリのパス

    Note:
        - 深度マップは0-255の範囲にスケーリングされて保存されます
        - 出力画像は元の画像と同じサイズにリサイズされます
        - 保存パスは {save_dir}/{ID}/{time_suffix}.png の形式になります
    """
    time_suffix = ["image_t", "image_t-0.5", "image_t-1.0"]
    for ID, images in batch:
        for t, (key, img_pil) in enumerate(images.items()):
            inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            semantic_path = f"{save_dir}/{ID}/{time_suffix[t]}.png"

            Path(semantic_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(formatted).save(semantic_path)


def get_max_depths_in_subregions(img_path: str, height: int = 20) -> list[float]:
    """画像を6つの領域に分割し、各領域の最大深度値を取得する

    Args:
        img_path (str): 深度画像のファイルパス
        height (int, optional): 解析対象とする画像上部のピクセル数。デフォルトは20。

    Returns:
        list[float]: 各領域の最大深度値のリスト。左から右の順に6つの値が格納される。
        各値は0-255の範囲の深度値を表す。

    Note:
        - 画像の上部analysis_heightピクセルのみを解析対象とする
        - 画像は水平方向に等間隔で6分割される
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    step = img.shape[1] / 6
    max_depths = {}
    for i in range(0, 6):
        x_start = round(i * step)
        x_end = round(x_start + step)
        chip = img[:height, x_start:x_end]
        max_depth = int(chip.max())
        max_depths[i] = max_depth
    return max_depths


def process_image_paths(img_paths: list[str]) -> pd.DataFrame:
    """画像パスのリストを処理し、各画像の最大深度値を含む辞書のリストを返す

    Args:
        img_paths (list[str]): 処理する画像パスのリスト

    Returns:
        list[dict]: 各画像に対する結果を含む辞書のリスト。
        各辞書には以下のキーが含まれる:
            - img_path: 画像の相対パス
            - max_depth_0 〜 max_depth_5: 6つの領域それぞれの最大深度値
    """
    results = []
    for img_path in tqdm(img_paths, total=len(img_paths)):
        max_depths = get_max_depths_in_subregions(img_path)
        img_path = f"{Path(img_path).parent.stem}/{Path(img_path).name}"
        row = {
            "path": img_path,
            "max_depth_0": max_depths[0],
            "max_depth_1": max_depths[1],
            "max_depth_2": max_depths[2],
            "max_depth_3": max_depths[3],
            "max_depth_4": max_depths[4],
            "max_depth_5": max_depths[5],
        }
        results.append(row)
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    save_dir = "/kaggle/dataset/depth"
    model_name = "Intel/dpt-large"
    depth_csv_path = "/kaggle/dataset/depth.csv"

    df_feature_train = pd.read_csv(TRAIN_FEATURES)
    df_feature_test = pd.read_csv(TEST__FEATURES)

    df_feature_train = add_image_paths(df_feature_train, IMAGE_DIR)
    df_feature_test = add_image_paths(df_feature_test, IMAGE_DIR)

    df_feature = pd.concat(
        [df_feature_train, df_feature_test], axis=0, ignore_index=True
    )

    processor = DPTImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name).to(DEVICE)

    dataset = SemanticSegmentationDataset(df_feature)
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda x: x,
        drop_last=False,
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        generate_semantic_maps(batch, model, processor, save_dir)

    img_paths = glob(f"{save_dir}/*/*.png")
    df = process_image_paths(img_paths)
    df.to_csv(depth_csv_path, index=False)
