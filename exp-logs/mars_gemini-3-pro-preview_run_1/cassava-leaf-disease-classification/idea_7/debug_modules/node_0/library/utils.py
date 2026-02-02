import os
import sys
import logging
import shutil
import torch
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy.
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


def get_logger(filename):
    """
    Creates a logger that writes to both a file and stdout.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)

        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def save_checkpoint(state, is_best, filepath=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str, optional): Path to save the checkpoint. Defaults to Config.OUTPUT_DIR/checkpoint.pth.
    """
    if filepath is None:
        filepath = os.path.join(Config.OUTPUT_DIR, "checkpoint.pth")

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, filepath, device=None):
    """
    Loads model weights from a checkpoint file.
    Handles 'module.' prefix removal if the model was trained with DataParallel.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for restoring optimizer/epoch), or None if not found.
    """
    if device is None:
        device = Config.DEVICE

    if not os.path.exists(filepath):
        # We return None so the caller can decide whether to error out or start from scratch
        return None

    checkpoint = torch.load(filepath, map_location=device)

    # Extract state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Fix potential DataParallel module prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v

    # Load into model
    model.load_state_dict(new_state_dict)

    return checkpoint


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model logits or probabilities.
        target (torch.Tensor): True labels.
        topk (tuple): Tuple of k values to compute accuracy for (e.g., (1, 5)).

    Returns:
        list: List of accuracy values (float) for each k.
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
