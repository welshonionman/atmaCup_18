import os
import warnings
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.constants import (
    TRAIN_FEATURES,
)
from src.image.test import inference
from src.image.train import training
from src.preprocess import make_train_folds
from src.util import set_seed

warnings.filterwarnings("ignore")


class CFG:
    timm_model_name = "tf_efficientnet_b7_ns"

    # ============== comp exp name =============
    exp_name = timm_model_name
    comp_dataset_path = "/kaggle/datasets/"

    oof_df_path = f"/kaggle/outputs/{exp_name}/submit/oof_cv.csv"
    submit_dir = f"/kaggle/outputs/{exp_name}/submit/"
    model_dir = f"/kaggle/outputs/{exp_name}/model/"
    fold_csv = f"/kaggle/dataset/train_folds.csv"

    is_debug = False
    use_gray_scale = False

    model_in_chans = 9  # モデルの入力チャンネル数

    # ============== model cfg =============
    model_name = timm_model_name

    num_frames = 3
    norm_in_chans = 1 if use_gray_scale else 3

    use_ema = False
    ema_decay = 0.995

    # ============== training cfg =============
    size = 224
    batch_size = 32

    use_amp = True

    scheduler = "GradualWarmupSchedulerV2"
    epochs = 20
    if is_debug:
        epochs = 1

    warmup_factor = 10
    lr = 1e-3
    if scheduler == "GradualWarmupSchedulerV2":
        lr /= warmup_factor

    # ============== fold =============
    n_fold = 5

    # ============== ほぼ固定 =============
    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    num_workers = 4

    # ============== augmentation =============
    train_aug_list = [
        A.Resize(size, size),
        # A.RandomBrightnessContrast(p=0.5),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.Normalize(
            mean=[0] * norm_in_chans * num_frames,
            std=[1] * norm_in_chans * num_frames,
        ),
        ToTensorV2(),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * norm_in_chans * num_frames,
            std=[1] * norm_in_chans * num_frames,
        ),
        ToTensorV2(),
    ]


if __name__ == "__main__":
    cfg = CFG()
    make_train_folds(TRAIN_FEATURES, cfg.fold_csv, cfg.n_fold)

    for dir_path in [
        Path(CFG.model_dir),
        Path(CFG.submit_dir),
    ]:
        os.makedirs(Path(dir_path), exist_ok=True)

    set_seed()
    training(cfg)
    inference(cfg)
