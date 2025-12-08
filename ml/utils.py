import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def prepare_device() -> torch.device:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logger.info("Using device: %s", device)
    return device


def convert_paths(obj):
    if isinstance(obj, Path):
        return str(obj)

    # Traverse dicts
    if isinstance(obj, dict):
        return {convert_paths(k): convert_paths(v) for k, v in obj.items()}

    # Traverse lists / tuples / sets
    if isinstance(obj, list):
        return [convert_paths(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_paths(x) for x in obj)
    if isinstance(obj, set):
        return {convert_paths(x) for x in obj}

    # Leave everything else untouched
    return obj

def conf_to_obj(cfg):
    obj = OmegaConf.to_container(cfg, resolve=True)
    # Convert PosixPath -> str
    obj = convert_paths(obj)
    return obj
