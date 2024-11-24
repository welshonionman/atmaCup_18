import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
from timm.utils.model_ema import ModelEmaV2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    DEVICE,
    TARGET_COLUMNS,
)
from src.image.dataset import CustomDataset, get_image_paths, load_train_data
from src.image.model import CustomModel
from src.image.preprocess import make_video_cache
from src.image.scheduler import get_scheduler, scheduler_step
from src.metrics import get_result, mae
from src.util import AverageMeter


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    model_ema=None,
):
    model.train()
    scaler = amp.GradScaler(enabled=True, device=device)

    losses = AverageMeter()
    preds = []
    preds_labels = []
    global_step = 0

    for step, (images, labels) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"epoch: {epoch}"
    ):
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        with amp.autocast(str(device)):
            y_preds = model(images)

            if y_preds.size(1) == 1:
                y_preds = y_preds.view(-1)

            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        optimizer.zero_grad()
        global_step += 1

        preds.append(y_preds.detach().to("cpu").numpy())
        preds_labels.append(labels.detach().to("cpu").numpy())

    train_loss = losses.val
    train_avg_loss = losses.avg
    lr = scheduler.get_lr()[0]

    print(f"{train_loss = :.4e} {train_avg_loss = :.4e} {lr = :.6e}")
    predictions = np.concatenate(preds)
    labels = np.concatenate(preds_labels)
    return losses.avg, predictions, labels


def valid_fn(valid_loader, model, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    preds = []

    for step, (images, labels) in tqdm(
        enumerate(valid_loader), total=len(valid_loader), desc=f"epoch: {epoch}"
    ):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)

        if y_preds.size(1) == 1:
            y_preds = y_preds.view(-1)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.to("cpu").numpy())

    valid_loss = losses.val
    valid_avg_loss = losses.avg
    print(f"{valid_loss = :.4f} {valid_avg_loss = :.4f}")
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def save_model(model, save_path, model_ema, valid_preds, fold):
    if model_ema is not None:
        torch.save(
            {"model": model_ema.module.state_dict(), "preds": valid_preds},
            save_path,
        )
    else:
        torch.save(
            {"model": model.state_dict(), "preds": valid_preds},
            save_path,
        )


def train_fold(folds, fold, video_cache, cfg):
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_labels = train_folds[TARGET_COLUMNS].values
    valid_labels = valid_folds[TARGET_COLUMNS].values

    train_dataset = CustomDataset(
        train_folds,
        video_cache,
        labels=train_labels,
        transform=A.Compose(cfg.train_aug_list),
    )
    valid_dataset = CustomDataset(
        valid_folds,
        video_cache,
        labels=valid_labels,
        transform=A.Compose(cfg.valid_aug_list),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = CustomModel(cfg, pretrained=True)
    model.to(DEVICE)

    if cfg.use_ema:
        model_ema = ModelEmaV2(model, decay=cfg.ema_decay)
    else:
        model_ema = None

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_scheduler(cfg, optimizer)

    criterion = nn.L1Loss()

    best_score = np.inf

    for epoch in range(cfg.epochs):
        # train
        avg_loss, train_preds, train_labels_epoch = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            DEVICE,
            model_ema,
        )

        # eval
        if model_ema is not None:
            avg_val_loss, valid_preds = valid_fn(
                valid_loader, model_ema.module, criterion, DEVICE
            )
        else:
            avg_val_loss, valid_preds = valid_fn(
                valid_loader, model, criterion, epoch, DEVICE
            )

        scheduler_step(scheduler, avg_val_loss, epoch)

        # scoring
        score = mae(valid_labels, valid_preds)
        update_best = score < best_score

        if update_best:
            best_score = score

            print("SAVED")

            best_model_path = f"{cfg.model_dir}/{cfg.model_name}_fold{fold}_best.pth"
            save_model(model, best_model_path, model_ema, valid_preds, fold)

    last_model_path = f"{cfg.model_dir}/{cfg.model_name}_fold{fold}_last.pth"
    save_model(model, last_model_path, model_ema, valid_preds, fold)

    model_path = f"{cfg.model_dir}/{cfg.model_name}_fold{fold}_best.pth"
    check_point = torch.load(model_path, map_location=torch.device("cpu"))
    pred_cols = [f"pred_{i}" for i in range(len(TARGET_COLUMNS))]

    check_point_pred = check_point["preds"]

    if check_point_pred.ndim == 1:
        check_point_pred = check_point_pred.reshape(-1, len(TARGET_COLUMNS))

    valid_folds[pred_cols] = check_point_pred
    return valid_folds


def training(cfg):
    train = load_train_data(cfg.fold_csv)
    paths = get_image_paths(train["base_path"].values)
    video_cache = make_video_cache(paths)

    oof_df = pd.DataFrame()
    for fold in range(cfg.n_fold):
        print(f"\n\nfold {fold} start")
        _oof_df = train_fold(train, fold, video_cache, cfg)
        oof_df = pd.concat([oof_df, _oof_df])
        score = get_result(_oof_df)
        print(f"fold {fold} score: {score:<.4f}")

    oof_df = oof_df.sort_values("ori_idx").reset_index(drop=True)

    score = get_result(oof_df)
    print(f"oof score: {score:<.4f}")

    oof_df.to_csv(cfg.oof_df_path, index=False)
