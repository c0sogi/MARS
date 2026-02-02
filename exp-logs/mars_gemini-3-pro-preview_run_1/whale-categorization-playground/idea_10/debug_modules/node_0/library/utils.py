import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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
    Calculates the Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions (torch.Tensor or np.ndarray): Shape (N, 5) containing the top 5 predicted class indices.
        targets (torch.Tensor or np.ndarray): Shape (N,) containing the ground truth class indices.

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure targets are flat
    targets = targets.reshape(-1)

    n = len(targets)
    if n == 0:
        return 0.0

    score = 0.0
    for i in range(n):
        pred = predictions[i]
        target = targets[i]

        # Find indices where prediction matches target
        # np.where returns a tuple, we take the first element (array of indices)
        matches = np.where(pred == target)[0]

        if len(matches) > 0:
            # The rank is the index (0-based), so we add 1 for 1-based rank
            rank = matches[0]
            if rank < 5:
                score += 1.0 / (rank + 1)

    return score / n


def save_checkpoint(
    state, is_best, dir_path=Config.CHECKPOINT_DIR, filename="checkpoint.pth.tar"
):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer, etc.).
        is_best (bool): If True, copies this checkpoint to 'model_best.pth.tar'.
        dir_path (str): The directory to save the checkpoint in. Defaults to Config.CHECKPOINT_DIR.
        filename (str): The name of the checkpoint file.
    """
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(dir_path, "model_best.pth.tar")
        shutil.copyfile(filepath, best_filepath)
