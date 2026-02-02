import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    Sets seeds for python random, numpy, and torch (CPU & CUDA).
    Configures cuDNN to be deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def calc_map5(predictions, targets):
    """
    Calculates Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions: List of lists or torch.Tensor containing the predicted class indices.
                     Shape should be (N, K) where K >= 5.
        targets: List or torch.Tensor containing the ground truth class indices.
                 Shape should be (N,).

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy/list if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are indexable lists/arrays
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    score = 0.0
    n_samples = len(targets)

    if n_samples == 0:
        return 0.0

    for preds, target in zip(predictions, targets):
        # We strictly evaluate the top 5 predictions
        top_preds = preds[:5]

        if target in top_preds:
            # Rank is 0-indexed in the list, so we add 1 for the metric
            rank = top_preds.index(target) + 1
            score += 1.0 / rank

    return score / n_samples
