import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from dataset import create_dataloaders
from net import Model
from train import train_model
from test import test
from dataclasses import dataclass
from typing import Optional
import record
import processing

@dataclass
class TrainConfig:
    mode: str = "train"
    dataset: str = "../ABCD_smoothed.npz"

@dataclass
class TestConfig:
    mode: str = "test"
    weights: Optional[str] = None
    dataset: str = "../ABCD_smoothed.npz"

@dataclass
class InferenceConfig:
    mode: str = "inference"
    weights: Optional[str] = None

@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Using device: {device}\n")

    model = Model(num_classes=4)
    model.to(device)

    if 'weights' in cfg:
        print(f"Loading model from {cfg.weights}")
        best_model = torch.load(cfg.weights, map_location=device)
        model.load_state_dict(best_model['model_state_dict'])

    if cfg.mode == "train":
        train_loader, val_loader, test_loader = create_dataloaders(
            cfg.dataset,
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
        test_acc, test_preds, test_labels = test(model, test_loader, device, num_classes=4)
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'test_accuracy': test_acc
        }, 'best_model.pth')
        print("Model saved to 'best_model.pth'")

    if cfg.mode == "test":
        train_loader, val_loader, test_loader = create_dataloaders(
                cfg.dataset,
                batch_size=128,
                val_ratio=0.10,
                test_ratio=0.20,
                seed=42,
            )   
        test_acc, test_preds, test_labels = test(model, test_loader, device, num_classes=4)

    if cfg.mode == "inference":
        for record_idx in range(1000):
            record.countdown("recording", 3)
            x = record.record(132, device)
            # Remove gyro channels, as we don't use them
            x = x[:3, :]
            # Apply smoothing before sending to model
            x = processing.process_raw(x)
            # add batch dim
            x = x.unsqueeze(dim=0)
            
            y = model(x)
            y_prob = F.softmax(y, dim=-1)

            print(f"P(A) = {y_prob[0, 0]}")
            print(f"P(B) = {y_prob[0, 1]}")
            print(f"P(C) = {y_prob[0, 2]}")
            print(f"P(D) = {y_prob[0, 3]}")

    

if __name__ == "__main__":
    main()
