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
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
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


def calculate_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model output logits or probabilities (N, C).
        target (torch.Tensor): Ground truth labels (N).
        topk (tuple): Tuple of k values for Top-K accuracy.

    Returns:
        list: A list of accuracy values (float) for each k in topk.
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
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.WORKING_DIR, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, optimizer=None, filename="model_best.pth"):
    """
    Loads a checkpoint from the working directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Name of the checkpoint file to load.

    Returns:
        int: The epoch number to resume from, or 0 if no checkpoint found/loaded.
        float: The best metric value (e.g. accuracy) if available, else 0.0.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    start_epoch = 0
    best_acc = 0.0

    if os.path.isfile(filepath):
        # Load to the configured device
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)

        # Load model state
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # Extract metadata
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)

        # print(f"Loaded checkpoint '{filename}' (epoch {checkpoint.get('epoch', 0)})")
    else:
        # print(f"No checkpoint found at '{filepath}'")
        pass

    return start_epoch, best_acc
