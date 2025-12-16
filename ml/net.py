import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DepthwiseSeparableBlock(nn.Module):
    """
    Lightweight Conv1d block that keeps parameter count tiny:
    depthwise (groups=in_channels) + pointwise projection.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm1d(in_channels)

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        return self.act(x)


class Model(nn.Module):
    """
    Tiny 1D CNN designed to fit comfortably under 128 kB when int8-quantized.
    Uses only depthwise-separable convolutions and a simple strided downsample.
    """

    def __init__(self, in_channels: int, num_classes: int, downsample_factor: int = 2):
        super().__init__()
        self.downsample_factor = max(1, int(downsample_factor))

        c1, c2, c3, c4 = 16, 24, 32, 48

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(c1, c2, kernel_size=5, stride=1),
            DepthwiseSeparableBlock(c2, c3, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(c3, c4, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(c4, c4, kernel_size=5, stride=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c4, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple temporal decimation to shrink compute; microcontroller-friendly (stride select).
        if self.downsample_factor > 1:
            x = x[..., :: self.downsample_factor]

        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    weights: Optional[Path]
    downsample_factor: int = 2


def load_model(cfg: ModelConfig, device: torch.device) -> Model:
    model = Model(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        downsample_factor=getattr(cfg, "downsample_factor", 2),
    )
    model.to(device)

    if cfg.weights is not None:
        logger.info("Loading model from %s", cfg.weights)
        weights = torch.load(cfg.weights, map_location=device)
        model.load_state_dict(weights)

    return model
