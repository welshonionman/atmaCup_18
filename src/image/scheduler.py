import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


def get_scheduler(cfg, optimizer):
    if cfg.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.factor,
            patience=cfg.patience,
            verbose=True,
            eps=cfg.eps,
        )
    elif cfg.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr, last_epoch=-1
        )
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1
        )
    elif cfg.scheduler == "GradualWarmupSchedulerV2":
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.epochs, eta_min=1e-7
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine
        )

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()
    elif isinstance(scheduler, GradualWarmupSchedulerV2):
        scheduler.step(epoch)
