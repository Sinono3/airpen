import logging
import subprocess
import time

import einops
import torch
from tqdm import tqdm
import serial

logger = logging.getLogger(__name__)

# --- Existing Constants ---
SERIAL_PORT = "/dev/cu.usbmodem1101"
SERIAL_BAUD = 115200
SERIAL_DATA_CMD = ["pio", "device", "monitor", "--port", SERIAL_PORT, "--quiet", "--baud", str(SERIAL_BAUD)]
# DEBUG:
# SERIAL_DATA_CMD = ["cat", "../test.csv"]

# --- Existing Functions (run_command_until_n_lines, record, countdown) are omitted for brevity, but they remain. ---

# ... [run_command_until_n_lines, record, countdown functions remain here] ...

def record_triggered(n_samples: int, device: torch.device | str, trigger_code: str = "INFERENCE") -> torch.Tensor:
    """
    Waits for a specific trigger code via serial, then receives n_samples of data.
    return shape: (6, n_samples)
    6 channels: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z (timestamp is ignored/removed)
    """
    x = []
    
    logger.info(f"Waiting for trigger code: '{trigger_code}' on {SERIAL_PORT}...")
    
    try:
        # Initialize serial connection
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        ser.flushInput() # Clear any junk in the buffer
        
        # 1. Wait for the trigger code
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line == trigger_code:
                logger.info("Trigger code received. Starting data collection...")
                break
            elif line:
                # Log any other serial output while waiting
                logger.debug(f"Received serial data while waiting: {line}")
            
            # Optional: Add a short sleep to prevent busy-waiting
            time.sleep(0.01)

        # 2. Receive N samples
        # Create progress bar
        with tqdm(total=n_samples, desc="Collecting samples", unit="sample") as pbar:
            while len(x) < n_samples:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if not line:
                        continue # Skip empty lines

                    # Data format: timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
                    sample_data = list(map(float, line.split(",")))
                    
                    if len(sample_data) == 7:
                        x.append(sample_data)
                        pbar.update(1)
                    else:
                        logger.warning(f"Skipping malformed data line: {line}")

                except ValueError:
                    logger.warning(f"Skipping non-numeric data line: {line}")
                except Exception as e:
                    logger.error(f"Error during data collection: {e}")
                    break
        
        ser.close()

    except serial.SerialException as e:
        logger.error(f"Serial Port Error: {e}")
        raise

    if len(x) < n_samples:
        logger.warning(f"Only collected {len(x)} samples instead of the expected {n_samples}.")

    # 3. Process the collected data
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    # Rearrange to (channels, time) -> (c, t)
    x_tensor = einops.rearrange(x_tensor, 't c -> c t')
    # remove timestamp (assumed to be channel 0)
    x_tensor = x_tensor[1:] 
    
    return x_tensor
# returns list of lines
def run_command_until_n_lines(command, n_lines) -> list[str] :
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=isinstance(command, str)
    )

    lines = []

    # Create progress bar
    with tqdm(total=n_lines, desc="Collecting output", unit="line") as pbar:
        for line in process.stdout:
            lines.append(line)
            pbar.update(1)
    
            if len(lines) >= n_lines:
                break

    if process.poll() is None:
        process.terminate()
        process.wait()

    return lines

def record(t, device: torch.device | str):
    """
    return shape: (7, t)
    7 channels: timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
    """
    x = []

    for line in run_command_until_n_lines(SERIAL_DATA_CMD, t+2):
        x.append(list(map(float, line.split(","))))

    x = x[1:-1]
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x = einops.rearrange(x, 't c -> c t')
    # remove timestamp
    x = x[1:]
    return x


def countdown(gerund: str, seconds: int):
    logger.info("%s in...", gerund)
    for i in range(seconds):
        logger.info("%s... ", seconds - i)
        time.sleep(1)
    logger.info("%s.", gerund)
