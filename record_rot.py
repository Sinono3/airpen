import logging
from pathlib import Path
import einops
import numpy as np
import datetime
import ml.record
import torch
angle = 270

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("outputs/rotations") / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

for i in range(8):
    print(f"Sample {i}")
    ml.record.countdown("recording", 2)
    x = ml.record.record(500, torch.device('cpu'))
    x = x[1:]
    x = einops.rearrange(x, "channel time -> time channel")

    output_path = output_dir / f"rot{angle}_recording{i}.csv"
    np.savetxt(
        output_path, x.cpu().numpy(), delimiter=","
    )
