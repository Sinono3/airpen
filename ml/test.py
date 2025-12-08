import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import net
import numpy as np
import torch
import utils
from aim import Run
from dataset import create_dataloaders
from hydra.core.config_store import ConfigStore
from net import ModelConfig
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def test(model, loader, device, num_classes: int, aim_run: Optional[Run] = None):
    """Test the model and return detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", leave=False):
            x = x.to(device)  # (B, 3, 132)
            y = y.to(device)
            
            outputs = model(x)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Compute per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    logger.info("%s", "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("%s", "=" * 60)
    logger.info("Overall Accuracy: %.2f%%", accuracy)
    logger.info("Per-Class Accuracy:")

    for cls in range(num_classes):
        cls_mask = all_labels == cls
        if cls_mask.sum() > 0:
            cls_acc = 100. * (all_preds[cls_mask] == all_labels[cls_mask]).sum() / cls_mask.sum()
            logger.info("  Class %s: %.2f%% (%d samples)", cls, cls_acc, cls_mask.sum())
            if aim_run is not None:
                aim_run.track(cls_acc, name="accuracy", context={"phase": "test", "class": cls})

    if aim_run is not None:
        aim_run.track(accuracy, name="accuracy", context={"phase": "test"})

    logger.info("%s", "=" * 60)
    return accuracy, all_preds, all_labels

@dataclass
class TestConfig:
    model: ModelConfig
    dataset: Path

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)

@hydra.main(config_path="./configs/", config_name="test", version_base=None)
def main(cfg: TestConfig):
    utils.setup_logging(Path.cwd() / "test.log")
    logger.info("Hydra run directory: %s", Path.cwd())

    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)
    aim_run = Run(repo=str(Path.cwd()), experiment="test")
    aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)

    train_loader, val_loader, test_loader = create_dataloaders(
            cfg.dataset,
            batch_size=128,
            seed=42,
        )
    test_acc, test_preds, test_labels = test(
        model, test_loader, device, num_classes=cfg.model.num_classes, aim_run=aim_run
    )
    aim_run["metrics/test_accuracy"] = test_acc
    aim_run.close()

if __name__ == "__main__":
    main()
