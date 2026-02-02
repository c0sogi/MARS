import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Calculates the Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions (torch.Tensor, np.ndarray, or list): Shape (N, 5) containing
            the top 5 predicted class IDs for each sample.
        targets (torch.Tensor, np.ndarray, or list): Shape (N,) containing
            the ground truth class ID for each sample.

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Basic shape validation
    if len(predictions) != len(targets):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    score = 0.0
    n_samples = len(targets)

    for i in range(n_samples):
        pred_row = predictions[i]
        target = targets[i]

        # Find the rank of the target in the predictions
        # np.where returns a tuple of arrays, we take the first array
        matches = np.where(pred_row == target)[0]

        if len(matches) > 0:
            # Rank is 1-based index
            rank = matches[0] + 1
            if rank <= 5:
                score += 1.0 / rank

    return score / n_samples if n_samples > 0 else 0.0
