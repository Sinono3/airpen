import copy
import logging
import pathlib
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
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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
        run["metrics/best_val_acc"] = best_val_acc

    return model, history, best_val_acc

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
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    utils.setup_logging(run_dir / "train.log")
    logger.info("Hydra run directory: %s", run_dir)

    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)
    aim_run = Run(repo=str(run_dir), experiment="train")
    aim_run["hparams"] = utils.conf_to_obj(cfg)
    
    # Copy model architecture and config to file
    (run_dir / "model_architecture.txt").write_text(str(model))
    (run_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.dataset,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        transforms=random_xz_rotation,
    )

    model, history, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
        run=aim_run,
    )

    logger.info("Running final test on test set...")
    test_acc, test_preds, test_labels = test(
        model, test_loader, device, num_classes=cfg.model.num_classes, aim_run=aim_run
    )

    best_model_path = run_dir / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
        },
        best_model_path,
        
    )
    
    aim_run["artifacts/best_model_path"] = str(best_model_path)
    aim_run["metrics/test_accuracy"] = test_acc
    aim_run.close()
    logger.info("Artifacts written to %s", run_dir)

if __name__ == "__main__":
    main()
