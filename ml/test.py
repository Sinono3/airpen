from dataclasses import dataclass
from pathlib import Path

import hydra
import net
import numpy as np
import torch
import utils
from dataset import create_dataloaders
from hydra.core.config_store import ConfigStore
from net import ModelConfig
from tqdm import tqdm


def test(model, loader, device, num_classes=4):
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
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-Class Accuracy:")
    
    for cls in range(num_classes):
        cls_mask = all_labels == cls
        if cls_mask.sum() > 0:
            cls_acc = 100. * (all_preds[cls_mask] == all_labels[cls_mask]).sum() / cls_mask.sum()
            print(f"  Class {cls}: {cls_acc:.2f}% ({cls_mask.sum()} samples)")
    
    print(f"{'='*60}\n")
    return accuracy, all_preds, all_labels

@dataclass
class TestConfig:
    model: ModelConfig
    dataset: Path

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)

@hydra.main(config_path="./configs/", config_name="test", version_base=None)
def main(cfg: TestConfig):
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
    LABELS = LABELS[:cfg.model.num_classes]
    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)

    train_loader, val_loader, test_loader = create_dataloaders(
            cfg.dataset,
            batch_size=128,
            val_ratio=0.10,
            test_ratio=0.20,
            seed=42,
        )   
    test_acc, test_preds, test_labels = test(model, test_loader, device, num_classes=4)   

if __name__ == "__main__":
    main()
