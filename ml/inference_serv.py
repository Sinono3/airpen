import logging
from dataclasses import dataclass

import time
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
import serial # <-- NEW DEPENDENCY

logger = logging.getLogger(__name__)

# Define the number of samples you expect
N_SAMPLES = 500

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
    
    # 1. Setup Serial Communication for Output
    try:
        # Use the same port and baud rate as defined in record.py
        ser_out = serial.Serial(record.SERIAL_PORT, record.SERIAL_BAUD, timeout=1)
        ser_out.flushOutput()
        logger.info(f"Serial output ready on {record.SERIAL_PORT} at {record.SERIAL_BAUD} baud.")
    except serial.SerialException as e:
        logger.error(f"Could not open serial port for output: {e}")
        return # Exit if serial output fails

    # Loop indefinitely, waiting for the trigger code
    for record_idx in range(1000):
        # 2. Wait for the "INFERENCE" trigger and receive N_SAMPLES
        try:
            x = record.record_triggered(N_SAMPLES, device)
        except Exception as e:
            logger.error(f"Failed to record data. Re-initializing serial wait loop: {e}")
            # Close and re-open the serial port if there was a problem
            ser_out.close()
            time.sleep(1) # Wait a moment
            try:
                ser_out = serial.Serial(record.SERIAL_PORT, record.SERIAL_BAUD, timeout=1)
                ser_out.flushOutput()
            except serial.SerialException as re:
                logger.error(f"Could not re-open serial port: {re}")
                break
            continue # Go to the next loop iteration (wait again)

        # Ensure we have data before proceeding
        if x.shape[1] < N_SAMPLES:
            logger.warning("Skipping inference due to insufficient samples collected.")
            continue
            
        # --- Existing Pre-processing Steps ---
        x = processing.smooth(x)
        x = processing.pca_transform_3_handedness(x[:3])
        x = processing.align_to_first_movement(x)
        x = x.unsqueeze(dim=0)

        # 3. Perform Inference
        with torch.no_grad():
            y = model(x)

        y_prob = F.softmax(y, dim=-1).mean(dim=0)  # Average predictions
        cls_idx = torch.argmax(y_prob[:], dim=-1)
        pred = LABELS[cls_idx]
        logger.info("predicted label: %s (confidence: %.3f)", pred, y_prob[cls_idx])
        for i, label in enumerate(LABELS):
            logger.info("P(%s) = %.4f", label, y_prob[i])
            
        # 4. Write predicted label to serial output
        output_data = f"PREDICTED_LABEL:{pred}\n"
        try:
            ser_out.write(output_data.encode('utf-8'))
            logger.info(f"Wrote prediction to serial: {pred}")
        except Exception as e:
            logger.error(f"Failed to write to serial port: {e}")

    # Ensure serial port is closed when the script finishes (e.g., after 1000 iterations)
    ser_out.close()


if __name__ == "__main__":
    main()
