from dataclasses import dataclass

import hydra
import net
import processing
import record
import torch
import torch.nn.functional as F
import utils
from hydra.core.config_store import ConfigStore
from net import ModelConfig


@dataclass
class InferenceConfig:
    model: ModelConfig

cs = ConfigStore.instance()
cs.store(name="inference", node=InferenceConfig)

@hydra.main(config_path="./configs/", config_name="inference", version_base=None)
def main(cfg: InferenceConfig):
    LABELS = ['A', 'B', 'C' , 'D', 'E', 'F']
    LABELS = LABELS[:cfg.model.num_classes]
    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)
    model.eval()

    for record_idx in range(1000):
        record.countdown("recording", 1)
        x = record.record(132, device)

        # Remove gyro channels, as we don't use them
        x = x[:3, :]
        # remove Y
        x = x[[0,2], :]
        # Apply smoothing before sending to model
        x = processing.process_raw(x)
        # add batch dim
        x = x.unsqueeze(dim=0)

        # normalize local to sample
        eps = 1e-10
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + eps)
        
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
