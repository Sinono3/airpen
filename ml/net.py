from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (B, 32, 66)
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (B, 64, 33)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)
            nn.Flatten(),             # -> (B, 128)
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

@dataclass
class ModelConfig:
    num_classes: int
    weights: Path

cs = ConfigStore.instance()
cs.store(group="model", name="base", node=ModelConfig)

def load_model(cfg: ModelConfig, device: torch.device | str) -> Model:
    model = Model(num_classes=cfg.num_classes)
    model.to(device)

    if cfg.weights is not None:
        print(f"Loading model from {cfg.weights}")
        best_model = torch.load(cfg.weights, map_location=device)
        model.load_state_dict(best_model['model_state_dict'])

    return model
