import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

# class Model(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         return self.net(x)

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.downsample = stride > 1 or in_channels != out_channels
        
        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut path (if dimensions change)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return self.relu(x)

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Initial Stem: Larger kernel (7) to capture immediate context
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Backbone: Residual Stages
        # Input to backbone is roughly (64, 125) due to stem stride/pool
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Crucial for regularization
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock1d(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x)

@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    weights: Optional[Path]

cs = ConfigStore.instance()
cs.store(group="model", name="base", node=ModelConfig)

def load_model(cfg: ModelConfig, device: torch.device | str) -> Model:
    model = Model(in_channels=cfg.in_channels, num_classes=cfg.num_classes)
    model.to(device)

    if cfg.weights is not None:
        logger.info("Loading model from %s", cfg.weights)
        weights = torch.load(cfg.weights, map_location=device)
        model.load_state_dict(weights)

    return model
