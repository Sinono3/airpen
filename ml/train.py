import copy
import inspect
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import net
import torch
import torch.nn as nn
import utils
from aim import Run
from dataset import create_dataloaders
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from net import Model, ModelConfig
from omegaconf import OmegaConf
from processing import random_xz_rotation
from test import test
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(model: Model, loader: DataLoader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)  # (B, 3, 132)
        y = y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    """Validate and return average loss and accuracy"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(device)  # (B, 3, 132)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_model(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: torch.device | str = "cuda",
    run: Optional[Run] = None,
) -> tuple[Model, dict, float]:
    """Main training function with validation"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    logger.info("Starting training on %s", device)
    logger.info("Training samples: %d | Validation samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        logger.info(
            "Epoch %d/%d | lr=%.5f | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% | best_val_acc=%.2f%%",
            epoch + 1,
            num_epochs,
            current_lr,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_val_acc,
        )

        if run is not None:
            run.track(train_loss, name="loss", step=epoch, context={"phase": "train"})
            run.track(train_acc, name="accuracy", step=epoch, context={"phase": "train"})
            run.track(val_loss, name="loss", step=epoch, context={"phase": "val"})
            run.track(val_acc, name="accuracy", step=epoch, context={"phase": "val"})
            run.track(current_lr, name="lr", step=epoch)

    model.load_state_dict(best_model_state)
    logger.info("Training completed. Best validation accuracy: %.2f%%", best_val_acc)
    if run is not None:
        run["best_val_acc"] = best_val_acc

    return model, best_val_acc

@dataclass
class TrainConfig:
    model: ModelConfig
    dataset: Path
    batch_size: int
    num_epochs: int
    lr: float
    seed: int

cs = ConfigStore.instance()
cs.store(group="model", name="base", node=ModelConfig)
cs.store(name="train", node=TrainConfig)
    
@hydra.main(config_path="./configs/", config_name="train", version_base=None)
def main(cfg: TrainConfig):
    root = Path.cwd()
    run = Run(repo=str(root), experiment="train")
    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)
    run["config"] = utils.conf_to_obj(cfg)
    run["sys_info"] = {
        "device": str(device)
    }
    run.set_artifacts_uri('file:///Users/aldo/Homework/Embedded/airpen/outputs/artifacts/')
    run.log_artifact(inspect.getsourcefile(main), "train.py")
    run.log_artifact(inspect.getsourcefile(net.Model), "net.py")
    run.log_artifact(inspect.getsourcefile(create_dataloaders), "dataset.py")

    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.dataset,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        transforms=random_xz_rotation,
    )

    model, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
        run=run,
    )

    logger.info("Running final test on test set...")
    test_acc, test_preds, test_labels = test(
        model, test_loader, device, num_classes=cfg.model.num_classes, run=run
    )

    run["test"] = {
        "acc": test_acc,
        "preds": test_preds.tolist(),
        "labels": test_labels.tolist(),
    }
    torch.save(model.state_dict(), "model.pth")
    run.log_artifact("model.pth")
    run.close()
    logger.info("Aim run closed. Checkpoint saved as artifact.")

if __name__ == "__main__":
    main()
