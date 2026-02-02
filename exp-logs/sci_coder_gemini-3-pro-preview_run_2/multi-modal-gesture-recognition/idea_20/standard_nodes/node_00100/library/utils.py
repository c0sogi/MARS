import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def get_device():
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        # print("Using device: CPU")
    return device


def ensure_dir(path):
    """
    Ensures that the directory exists. If not, creates it.

    Args:
        path (str): The directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training epochs.
    """

    def __init__(self, name, fmt=":f"):
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


def save_checkpoint(
    state, is_best, checkpoint_dir=Config.CACHE_DIR, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    ensure_dir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)
        # Also copy to submission directory as required for the final artifact
        submission_best_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
        shutil.copyfile(filepath, submission_best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): Device to map the storage to.

    Returns:
        int: The epoch to resume from (start_epoch).
        float: The best metric value (e.g., best_loss) if available.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    return start_epoch, best_metric


def log_metrics(epoch, metrics):
    """
    Prints metrics with full precision.

    Args:
        epoch (int): Current epoch number.
        metrics (dict): Dictionary of metric names and values.
    """
    msg = f"Epoch {epoch}: "
    for k, v in metrics.items():
        msg += f"{k}={v} "
    print(msg.strip())


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
