import os
import random
import shutil
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def calculate_map5(predictions, targets):
    """
    Computes the Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions (torch.Tensor or np.ndarray): Shape (N, 5) containing top 5 predicted class indices.
        targets (torch.Tensor or np.ndarray): Shape (N,) containing ground truth class indices.

    Returns:
        float: The MAP@5 score.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure predictions are top-5
    if predictions.shape[1] > 5:
        predictions = predictions[:, :5]

    n = len(targets)
    if n == 0:
        return 0.0

    score = 0.0
    for i in range(n):
        pred = predictions[i]
        target = targets[i]

        # In this specific task, there is only 1 ground truth label per image.
        # MAP@5 simplifies to 1/(rank+1) if target is in predictions, else 0.
        if target in pred:
            # np.where returns a tuple of arrays, we want the first index
            rank = np.where(pred == target)[0][0]
            score += 1.0 / (rank + 1.0)

    return score / n


def save_checkpoint(
    state, is_best, checkpoint_dir="./working", filename="checkpoint.pth.tar"
):
    """
    Saves the model state to a file.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        int: The epoch to resume from (if present in checkpoint), else 0.
        float: The best metric score (if present), else 0.0.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
