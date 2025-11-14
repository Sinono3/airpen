import torch

def prepare_device() -> torch.device:
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Using device: {device}\n")
