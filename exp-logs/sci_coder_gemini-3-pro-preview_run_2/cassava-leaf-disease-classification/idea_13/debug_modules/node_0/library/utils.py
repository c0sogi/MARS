import os
import sys
import random
import logging
import numpy as np
import torch
import shutil
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str = None):
    """
    Creates and configures a logger that outputs to both stdout and a file.

    Args:
        log_file (str): Path to the log file. If None, uses default from Config.
    """
    if log_file is None:
        log_file = os.path.join(Config.LOG_DIR, "train.log")

    logger = logging.getLogger(Config.EXPERIMENT_NAME)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to set up file logging: {e}")

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
    """

    def __init__(self, name: str = "Metric", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


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


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current fold number.
    """
    filename = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )
        shutil.copyfile(filename, best_filename)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # Handle case where model was saved with DataParallel (keys start with 'module.')
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
