import timm
import torch.nn as nn
import numpy as np
from src.constants import TARGET_COLUMNS


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False, model_name=None):
        super().__init__()
        model_name = cfg.model_name

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=cfg.model_in_chans,
        )

        self.n_features = self.model.num_features

        self.target_size = len(TARGET_COLUMNS)

        self.fc = nn.Sequential(nn.Linear(self.n_features, self.target_size))

    def forward(self, image):
        feature = self.model(image)
        output = self.fc(feature)
        return output


class EnsembleModel:
    def __init__(self):
        self.models = []

    def __call__(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x).to("cpu").numpy())

        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)
