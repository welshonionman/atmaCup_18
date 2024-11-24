import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from src.constants import DEVICE, IMAGE_DIR, TEST__FEATURES, TRAIN_FEATURES
from src.feature_extract.common import SemanticSegmentationDataset, add_image_paths


def generate_semantic_maps(
    batch: list[tuple[str, dict[str, Image.Image]]],
    model: AutoModelForSemanticSegmentation,
    processor: AutoImageProcessor,
    save_dir: str,
):
    """セマンティックセグメンテーションモデルを使用して画像バッチのセグメンテーションマップを生成し保存する関数

    バッチ内の各画像に対してセグメンテーションを行い、結果を画像として保存します。
    各画像は3つの時点(t, t-0.5, t-1.0)について処理されます。

    Args:
        batch (list[tuple[str, dict[str, Image.Image]]]): 処理する画像バッチ。
            各要素は(ID, images)のタプルで、IDは画像の識別子、imagesは時点ごとの画像を含む辞書
        model (AutoModelForSemanticSegmentation): セグメンテーションモデル
        processor (AutoImageProcessor): 画像の前処理を行うプロセッサ
        save_dir (str): 生成されたセグメンテーションマップを保存するディレクトリのパス

    Note:
        - セグメンテーション結果は各ピクセルにクラスIDが割り当てられた画像として保存されます
        - 出力画像は元の画像と同じサイズにリサイズされます
        - 保存パスは {save_dir}/{ID}/{time_suffix}.png の形式になります
    """
    time_suffix = ["image_t", "image_t-0.5", "image_t-1.0"]
    for ID, images in batch:
        for t, (key, img_pil) in enumerate(images.items()):
            inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                prediction = torch.nn.functional.interpolate(
                    logits, size=img_pil.size[::-1], mode="bicubic", align_corners=False
                )

                prediction = torch.argmax(prediction, dim=1)
                prediction = prediction.squeeze().cpu().numpy()

            semantic_path = f"{save_dir}/{ID}/{time_suffix[t]}.png"

            Path(semantic_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(semantic_path, prediction)


def merge_vehicle_labels(id2label: dict) -> dict:
    """車両関係のラベルをcarに統一"""
    vehicle_labels = {
        "20": "car",  # car (基準)
        "80": "car",  # bus
        "83": "car",  # truck
        "102": "car",  # van
        "103": "car",  # ship
        "116": "car",  # minibike
        "127": "car",  # bicycle
    }

    new_id2label = id2label.copy()

    for vehicle_id in vehicle_labels:
        if vehicle_id in new_id2label:
            new_id2label[vehicle_id] = "car"

    return new_id2label


def merge_tree_labels(id2label: dict) -> dict:
    """植物関係のラベルをtreeに統一"""
    tree_labels = {
        "4": "tree",  # tree (基準)
        "72": "tree",  # palm
        "17": "tree",  # plant
        "66": "tree",  # flower
    }

    new_id2label = id2label.copy()

    for tree_id in tree_labels:
        if tree_id in new_id2label:
            new_id2label[tree_id] = "tree"

    return new_id2label


def count_pixels_per_label(
    img_path: str, id2label: dict, target_labels: list[str]
) -> dict[str, int]:
    """画像内の各ラベルのピクセル数をカウントする関数

    セグメンテーション画像を読み込み、指定されたラベルごとのピクセル数をカウントします。
    画像内の各ピクセル値をid2labelを使ってラベルに変換し、target_labelsに含まれるラベルの
    ピクセル数をカウントして返します。

    Args:
        img_path (str): セグメンテーション画像のファイルパス
        id2label (dict[str, str]): ピクセル値(str)からラベル名(str)への変換辞書
        target_labels (list[str]): カウント対象のラベル名のリスト

    Returns:
        dict[str, int]: key:ラベル名、value:ピクセル数の辞書
                       target_labelsに含まれるラベルのみが含まれます
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    unique, counts = np.unique(img, return_counts=True)

    label2count = {label: 0 for label in target_labels}
    label2count["path"] = f"{Path(img_path).parent.stem}/{Path(img_path).name}"

    for pixel_value, count in zip(unique, counts, strict=True):
        pixel_value = str(pixel_value)
        if pixel_value in id2label:
            label = id2label[pixel_value]
            if label in target_labels:
                label2count[label] += count

    return label2count


# https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/blob/main/config.json
id2label = {
    "0": "wall",
    "1": "building",
    "2": "sky",
    "3": "floor",
    "4": "tree",
    "5": "ceiling",
    "6": "road",
    "7": "bed ",
    "8": "windowpane",
    "9": "grass",
    "10": "cabinet",
    "11": "sidewalk",
    "12": "person",
    "13": "earth",
    "14": "door",
    "15": "table",
    "16": "mountain",
    "17": "plant",
    "18": "curtain",
    "19": "chair",
    "20": "car",
    "21": "water",
    "22": "painting",
    "23": "sofa",
    "24": "shelf",
    "25": "house",
    "26": "sea",
    "27": "mirror",
    "28": "rug",
    "29": "field",
    "30": "armchair",
    "31": "seat",
    "32": "fence",
    "33": "desk",
    "34": "rock",
    "35": "wardrobe",
    "36": "lamp",
    "37": "bathtub",
    "38": "railing",
    "39": "cushion",
    "40": "base",
    "41": "box",
    "42": "column",
    "43": "signboard",
    "44": "chest of drawers",
    "45": "counter",
    "46": "sand",
    "47": "sink",
    "48": "skyscraper",
    "49": "fireplace",
    "50": "refrigerator",
    "51": "grandstand",
    "52": "path",
    "53": "stairs",
    "54": "runway",
    "55": "case",
    "56": "pool table",
    "57": "pillow",
    "58": "screen door",
    "59": "stairway",
    "60": "river",
    "61": "bridge",
    "62": "bookcase",
    "63": "blind",
    "64": "coffee table",
    "65": "toilet",
    "66": "flower",
    "67": "book",
    "68": "hill",
    "69": "bench",
    "70": "countertop",
    "71": "stove",
    "72": "palm",
    "73": "kitchen island",
    "74": "computer",
    "75": "swivel chair",
    "76": "boat",
    "77": "bar",
    "78": "arcade machine",
    "79": "hovel",
    "80": "bus",
    "81": "towel",
    "82": "light",
    "83": "truck",
    "84": "tower",
    "85": "chandelier",
    "86": "awning",
    "87": "streetlight",
    "88": "booth",
    "89": "television receiver",
    "90": "airplane",
    "91": "dirt track",
    "92": "apparel",
    "93": "pole",
    "94": "land",
    "95": "bannister",
    "96": "escalator",
    "97": "ottoman",
    "98": "bottle",
    "99": "buffet",
    "100": "poster",
    "101": "stage",
    "102": "van",
    "103": "ship",
    "104": "fountain",
    "105": "conveyer belt",
    "106": "canopy",
    "107": "washer",
    "108": "plaything",
    "109": "swimming pool",
    "110": "stool",
    "111": "barrel",
    "112": "basket",
    "113": "waterfall",
    "114": "tent",
    "115": "bag",
    "116": "minibike",
    "117": "cradle",
    "118": "oven",
    "119": "ball",
    "120": "food",
    "121": "step",
    "122": "tank",
    "123": "trade name",
    "124": "microwave",
    "125": "pot",
    "126": "animal",
    "127": "bicycle",
    "128": "lake",
    "129": "dishwasher",
    "130": "screen",
    "131": "blanket",
    "132": "sculpture",
    "133": "hood",
    "134": "sconce",
    "135": "vase",
    "136": "traffic light",
    "137": "tray",
    "138": "ashcan",
    "139": "fan",
    "140": "pier",
    "141": "crt screen",
    "142": "plate",
    "143": "monitor",
    "144": "bulletin board",
    "145": "shower",
    "146": "radiator",
    "147": "glass",
    "148": "clock",
    "149": "flag",
}


if __name__ == "__main__":
    save_dir = "/kaggle/dataset/segformer"
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    count_pixels_per_label_path = "/kaggle/dataset/count_pixels_per_label.csv"

    df_feature_train = pd.read_csv(TRAIN_FEATURES)
    df_feature_test = pd.read_csv(TEST__FEATURES)

    df_feature_train = add_image_paths(df_feature_train, IMAGE_DIR)
    df_feature_test = add_image_paths(df_feature_test, IMAGE_DIR)

    df_feature = pd.concat(
        [df_feature_train, df_feature_test], axis=0, ignore_index=True
    )

    processor = AutoImageProcessor.from_pretrained(model_name)

    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(DEVICE)
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

    id2label = merge_vehicle_labels(id2label)
    id2label = merge_tree_labels(id2label)

    target_labels = [
        "road",
        "wall",
        "car",
        "building",
        "tree",
        "floor",
        "sky",
        "fence",
        "sidewalk",
        "grass",
        "person",
    ]

    df = pd.DataFrame(columns=["path"] + target_labels)

    imgs = glob(f"{save_dir}/*/*.png")
    for i_, img_path in tqdm(enumerate(imgs), total=len(imgs)):
        img_counts = count_pixels_per_label(img_path, id2label, target_labels)
        df.loc[i_] = img_counts

    df.to_csv(count_pixels_per_label_path, index=False)
