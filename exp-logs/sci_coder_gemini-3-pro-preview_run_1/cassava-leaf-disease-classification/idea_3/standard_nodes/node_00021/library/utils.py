import os
import sys
import random
import numpy as np
import torch
import shutil
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Worker init function for DataLoader to ensure deterministic data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(log_file):
    """
    Creates a logger that writes to both a file and stdout.
    """
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)

        # Create formatters and add to handlers
        formatter = logging.Formatter("%(message)s")
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint. If is_best is True, copies it to best_model.pth.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(output_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, path, device):
    """
    Loads model weights from a checkpoint file.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at '{path}'")
        return None

    checkpoint = torch.load(path, map_location=device)

    # Determine if the checkpoint is a dict with 'state_dict' key or just the weights
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel keys (strip 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"Loaded checkpoint '{path}'")

    return checkpoint
