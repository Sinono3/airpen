import logging
from dataclasses import dataclass

import einops
import hydra
import net
import numpy as np
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
        # gravity = torch.tensor(processing.GRAVITY, device=x.device)
        # x = processing.remove_gravity_constant(x)
        # x[:3] = processing.align_to_plane(x[:3], gravity)
        
        x = processing.smooth(x)
        x = processing.pca_transform_3_handedness(x[:3])
        x = processing.align_to_first_movement(x)
        
        # batch
        # x = torch.stack([
        #     x[[0, 1, 2], :],
        #     x[[0, 2, 1], :],
        #     x[[1, 0, 2], :],
        #     x[[1, 2, 0], :],
        #     x[[2, 1, 0], :],
        #     x[[2, 0, 1], :],
        # ])
        x = x.unsqueeze(dim=0)

        with torch.no_grad():
            y = model(x)

        y_prob = F.softmax(y, dim=-1).mean(dim=0)  # Average predictions
        cls_idx = torch.argmax(y_prob[:], dim=-1)
        pred = LABELS[cls_idx]
        logger.info("predicted label: %s (confidence: %.3f)", pred, y_prob[cls_idx])
        for i, label in enumerate(LABELS):
            logger.info("P(%s) = %.4f", label, y_prob[i])

if __name__ == "__main__":
    main()
