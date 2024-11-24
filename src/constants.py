import numpy as np
import torch

IMAGE_DIR = "/kaggle/input/atmaCup#18_dataset/images/"
LIGHT_DIR = "/kaggle/input/atmaCup#18_dataset/traffic_lights/"

TRAIN_FEATURES = "/kaggle/input/atmaCup#18_dataset/train_features.csv"
TEST__FEATURES = "/kaggle/input/atmaCup#18_dataset/test_features.csv"

INTRINSIC_MATRIX = np.array(
    [[226.16438356, 0.0, 63.62426614], [0.0, 224.82352941, 11.76], [0.0, 0.0, 1.0]]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_NAMES = [
    "image_t-1.0.png",
    "image_t-0.5.png",
    "image_t.png",
]

TARGET_COLUMNS = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "x_2",
    "y_2",
    "z_2",
    "x_3",
    "y_3",
    "z_3",
    "x_4",
    "y_4",
    "z_4",
    "x_5",
    "y_5",
    "z_5",
]

TRAFFIC_CLASS = [
    "green",
    "straight",
    "left",
    "right",
    "empty",
    "other",
    "yellow",
    "red",
]
