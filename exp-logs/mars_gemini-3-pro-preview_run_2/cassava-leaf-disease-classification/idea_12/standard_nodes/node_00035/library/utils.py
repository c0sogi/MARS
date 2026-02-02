import os
import sys
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark = False is safer for reproducibility, though potentially slower
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    """
    Initializes and returns a logger that outputs to both a file and stdout.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File Handler
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy.
    """

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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, output_dir=None, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.
    If is_best is True, copies the checkpoint to 'model_best.pth'.
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(output_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, path, device=None):
    """
    Loads model weights from a checkpoint file.
    Handles 'module.' prefix removal if the model was trained with DataParallel.
    """
    if device is None:
        device = Config.DEVICE

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    # Determine where the state dict is stored
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Create a new state dict without the 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return checkpoint
