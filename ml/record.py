import subprocess
import time
import torch
import einops
from tqdm import tqdm

SERIAL_DATA_CMD = ["pio", "device", "monitor", "--quiet", "--baud", "115200"]
# DEBUG:
# SERIAL_DATA_CMD = ["cat", "../test.csv"]

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
    return shape: (6, t)
    6 channels: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
    """
    x = []

    for line in run_command_until_n_lines(SERIAL_DATA_CMD, t):
        x.append(list(map(float, line.split(","))))

    x = torch.tensor(x, dtype=torch.float32, device=device)
    x = einops.rearrange(x, 't c -> c t') # c = 6
    return x


def countdown(gerund: str, seconds: int):
    print(f"{gerund} in...")
    for i in range(seconds):
        print(f"{seconds - i}... ")
        time.sleep(1)
    print(f"{gerund}.")
