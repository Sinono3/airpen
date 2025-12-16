from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass, field
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
from net import Model, ModelConfig
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


def create_optimizer(model: Model, optimizer_cfg):
    try:
        optimizer_cls = getattr(torch.optim, optimizer_cfg.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown optimizer: {optimizer_cfg.name}") from exc
    return optimizer_cls(model.parameters(), **optimizer_cfg.params)


def create_scheduler(optimizer, scheduler_cfg):
    if scheduler_cfg is None or scheduler_cfg.name is None:
        return None
    try:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cfg.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}") from exc
    return scheduler_cls(optimizer, **scheduler_cfg.params)


def step_scheduler(scheduler, metric):
    if scheduler is None:
        return
    try:
        scheduler.step(metric)
    except TypeError:
        scheduler.step()

def train_model(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    optimizer_cfg: OptimizerConfig | None = None,
    scheduler_cfg: SchedulerConfig | None = None,
    device: torch.device | str = "cuda",
    run: Optional[Run] = None,
) -> tuple[Model, float]:
    """Main training function with validation"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_cfg)
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    logger.info("Starting training on %s", device)
    logger.info("Training samples: %d | Validation samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    try:
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            step_scheduler(scheduler, val_loss)
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
    except KeyboardInterrupt:
        logger.error("Keyboard interrupt detected. Stopping training early")

    logger.info("Returning model with best validation accuracy.")
    model.load_state_dict(best_model_state)
    logger.info("Training completed. Best validation accuracy: %.2f%%", best_val_acc)
    if run is not None:
        run["best_val_acc"] = best_val_acc

    return model, best_val_acc

@dataclass
class OptimizerConfig:
    name: str = ""
    params: dict = field(default_factory=lambda: {})


@dataclass
class SchedulerConfig:
    name: str | None = ""
    params: dict = field(default_factory=lambda: {})


@dataclass
class TrainConfig:
    model: ModelConfig
    dataset: Path
    batch_size: int
    num_epochs: int
    seed: int
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig | None = SchedulerConfig()

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
        seed=cfg.seed
    )
    logger.info("Steps per epoch: %s", len(train_loader))

    model, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
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
    hash = run.hash
    run.close()
    logger.info(f"Aim run closed. Checkpoint saved as artifact at /Users/aldo/Homework/Embedded/airpen/outputs/artifacts/{hash}/model.pth")

if __name__ == "__main__":
    main()
