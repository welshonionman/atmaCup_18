import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def add_image_paths(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    time_points = ["img_path_t_00", "img_path_t_05", "img_path_t_10"]
    time_values = ["image_t", "image_t-0.5", "image_t-1.0"]

    for time_point, time_value in zip(time_points, time_values, strict=True):
        image_paths = []
        for id_value in df.ID:
            image_path = f"{image_dir}/{id_value}/{time_value}.png"
            image_paths.append(image_path)
        df[time_point] = image_paths

    return df


class SemanticSegmentationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        images = {}

        img_t_00 = Image.open(row.img_path_t_00).convert("RGB")
        img_t_05 = Image.open(row.img_path_t_05).convert("RGB")
        img_t_10 = Image.open(row.img_path_t_10).convert("RGB")

        images["img_t_0"] = img_t_00
        images["img_t_1"] = img_t_05
        images["img_t_2"] = img_t_10

        return row.ID, images
