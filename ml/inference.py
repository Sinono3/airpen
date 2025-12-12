import einops
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    model: ModelConfig

cs = ConfigStore.instance()
cs.store(name="inference", node=InferenceConfig)

@hydra.main(config_path="./configs/", config_name="inference", version_base=None)
def main(cfg: InferenceConfig):
    LABELS = [
        "A",
        "C",
        "E",
        "I",
        "L",
        "N",
        "O",
        "R",
        "S",
        "T",
    ]
    device = utils.prepare_device()
    model = net.load_model(cfg.model, device)
    model.eval()

    for record_idx in range(1000):
        record.countdown("recording", 1)
        x = record.record(500, device)

        # Remove timestamp
        x = x[1:]
        # Remove gyro channels, as we don't use them
        x = x[:3, :]
        # Apply smoothing before sending to model
        print (x.shape)
        x = processing.process_raw(x)
        X = []
        # for i in range(100):
        #     X.append(processing.random_rotation(x, normal=torch.tensor([0.0, 0.0, -1.0], device=torch.device('mps')), normal_std=6.14))
        X = x.unsqueeze(dim=0)
        x, _ = einops.pack(X, "* channel time")

        # # add batch dim
        # x = x.unsqueeze(dim=0)
        print(x.shape)

        # normalize local to sample
        eps = 1e-10
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + eps)
        
        with torch.no_grad():
            y = model(x)

        y_prob = F.softmax(y, dim=-1)

        cls_idx = torch.argmax(y_prob[0, :], dim=-1)
        pred = LABELS[cls_idx]
        logger.info("predicted label: %s (confidence: %.3f)", pred, y_prob[0, cls_idx])
        for i, label in enumerate(LABELS):
            logger.info("P(%s) = %.4f", label, y_prob[0, i])

if __name__ == "__main__":
    main()
