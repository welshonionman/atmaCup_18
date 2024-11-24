import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    DEVICE,
    IMAGE_DIR,
    TARGET_COLUMNS,
    TEST__FEATURES,
)
from src.image.dataset import CustomDataset
from src.image.model import CustomModel, EnsembleModel
from src.image.preprocess import make_video_cache


def test_fn(valid_loader, model, device):
    preds = []

    for step, (images) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        preds.append(y_preds)

    predictions = np.concatenate(preds)
    return predictions


def inference(cfg):
    test = pd.read_csv(TEST__FEATURES)

    test["base_path"] = IMAGE_DIR + test["ID"]

    paths = []
    for base_path in test["base_path"].values:
        suffixs = ["image_t-1.0.png", "image_t-0.5.png", "image_t.png"]
        for suffix in suffixs:
            path = f"{base_path}/{suffix}"
            paths.append(path)

    video_cache = make_video_cache(paths)

    valid_dataset = CustomDataset(
        test,
        video_cache,
        cfg,
        transform=A.Compose(cfg.valid_aug_list),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = EnsembleModel()
    folds = list(range(cfg.n_fold))
    for fold in folds:
        _model = CustomModel(cfg, pretrained=False)
        _model.to(DEVICE)

        model_path = f"{cfg.model_dir}/{cfg.model_name}_fold{fold}_best.pth"
        state = torch.load(model_path)["model"]
        _model.load_state_dict(state)
        _model.eval()
        model.add_model(_model)

    preds = test_fn(valid_loader, model, DEVICE)

    test[TARGET_COLUMNS] = preds
    test.to_csv(f"{cfg.submit_dir}/submission_oof.csv", index=False)
    test[TARGET_COLUMNS].to_csv(
        f"{cfg.submit_dir}/submission_{cfg.exp_name}.csv", index=False
    )
