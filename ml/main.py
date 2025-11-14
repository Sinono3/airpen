import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import torch
import torch.nn.functional as F
from dataset import create_dataloaders
from net import Model
from train import train_model
from test import test
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import record
import processing
import utils

class Mode(Enum):
    TRAIN = "train"
    TEST = "test"
    INFERENCE = "inference"

@dataclass
class ModelConfig:
    num_classes: int = 4
    weights: Optional[Path] = None

@dataclass
class TrainConfig:
    dataset: Path

@dataclass
class TestConfig:
    dataset: Path

@dataclass
class InferenceConfig:
    pass

@dataclass
class Config:
    mode: Mode
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig(dataset="")
    test: TestConfig = TestConfig(dataset="")
    inference: InferenceConfig = InferenceConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
    
@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: Config):
    device = utils.prepare_device()
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
    LABELS = LABELS[:cfg.model.num_classes]
    model = Model(num_classes=cfg.model.num_classes)
    model.to(device)

    if cfg.model.weights is not None:
        print(f"Loading model from {cfg.model.weights}")
        best_model = torch.load(cfg.model.weights, map_location=device)
        model.load_state_dict(best_model['model_state_dict'])

    if cfg.mode == Mode.TRAIN:
        train_loader, val_loader, test_loader = create_dataloaders(
            cfg.train.dataset,
            batch_size=128,
            val_ratio=0.10,
            test_ratio=0.20,
            seed=42,
        )
    
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=50,
            lr=0.001,
            device=device,
        )
    
        print("\nRunning final test on test set...")
        test_acc, test_preds, test_labels = test(
            model, test_loader, device, num_classes=cfg.model.num_classes
        )
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'test_accuracy': test_acc
        }, 'best_model.pth')
        print("Model saved to 'best_model.pth'")

    if cfg.mode == Mode.TEST:
        train_loader, val_loader, test_loader = create_dataloaders(
                cfg.test.dataset,
                batch_size=128,
                val_ratio=0.10,
                test_ratio=0.20,
                seed=42,
            )   
        test_acc, test_preds, test_labels = test(model, test_loader, device, num_classes=4)

    if cfg.mode == Mode.INFERENCE:
        model.eval()

        for record_idx in range(1000):
            record.countdown("recording", 1)
            x = record.record(132, device)

            # Remove gyro channels, as we don't use them
            x = x[:3, :]
            # Apply smoothing before sending to model
            x = processing.process_raw(x)
            # add batch dim
            x = x.unsqueeze(dim=0)
            
            with torch.no_grad():
                y = model(x)

            y_prob = F.softmax(y, dim=-1)

            cls_idx = torch.argmax(y_prob[0, :], dim=-1)
            pred = LABELS[cls_idx]
            print(f"predicted label: {pred} (confidence: {y_prob[0, cls_idx]:.3f})")
            for i, label in enumerate(LABELS):
                print(f"P({label}) = {y_prob[0, i]}")

    

if __name__ == "__main__":
    main()
