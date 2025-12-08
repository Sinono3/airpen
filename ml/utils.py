import logging
import sys
from pathlib import Path, PosixPath
from typing import Optional

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logging with console and file handlers.
    """
    log_file_path = Path(log_file) if log_file is not None else Path.cwd() / "run.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file_path for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.debug("Logging initialized at %s (file: %s)", logging.getLevelName(level), log_file_path)
    return root_logger


def prepare_device() -> torch.device:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logger.info("Using device: %s", device)
    return device


def convert_paths(obj):
    # Convert single PosixPath
    if isinstance(obj, PosixPath):
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
