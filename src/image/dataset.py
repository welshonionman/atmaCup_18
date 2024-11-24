import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import IMAGE_DIR, IMAGE_NAMES


def load_train_data(fold_csv_path: str) -> pd.DataFrame:
    train = pd.read_csv(fold_csv_path)
    train["ori_idx"] = train.index
    train["scene"] = train["ID"].str.split("_").str[0]
    train["base_path"] = IMAGE_DIR + train["ID"]
    return train


def get_image_paths(base_paths: list) -> list:
    """画像パスのリストを生成する

    Args:
        base_paths (list): ベースとなるパスのリスト

    Returns:
        list: 画像パスのリスト
    """
    paths = []
    for base_path in base_paths:
        suffixs = ["image_t-1.0.png", "image_t-0.5.png", "image_t.png"]
        for suffix in suffixs:
            path = f"{base_path}/{suffix}"
            paths.append(path)
    return paths


class CustomDataset(Dataset):
    def __init__(self, df, video_cache, labels=None, transform=None):
        self.df = df
        self.base_paths = df["base_path"].values
        self.video_cache = video_cache
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_image_multiframe(self, idx):
        base_path = self.base_paths[idx]
        images = []
        suffixs = IMAGE_NAMES
        for suffix in suffixs:
            path = f"{base_path}/{suffix}"
            image = self.video_cache[path]
            images.append(image)
        return images

    def __getitem__(self, idx):
        image = self.read_image_multiframe(idx)
        if self.transform:
            images = []
            for img in image:
                sample = self.transform(image=img)
                images.append(sample["image"])

            image = torch.concat(images, dim=0)

        if self.labels is None:
            return image

        label = torch.tensor(self.labels[idx]).float()

        return image, label
