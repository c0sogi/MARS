import os
import random
import numpy as np
import torch
from library.config import CFG


def seed_everything(seed=CFG.seed):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to CFG.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def map_at_k(predictions, targets, k=5):
    """
    Calculates the Mean Average Precision at K (MAP@K) for single-ground-truth tasks.

    Args:
        predictions (np.ndarray, list, or torch.Tensor):
            Predicted labels or IDs. Shape should be (N, M) where M >= k.
            Rows should be sorted by confidence (most confident first).
        targets (np.ndarray, list, or torch.Tensor):
            Ground truth labels or IDs. Shape should be (N,).
        k (int):
            The number of top predictions to consider.

    Returns:
        float: The calculated MAP@K score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    num_samples = len(targets)
    if num_samples == 0:
        return 0.0

    score_sum = 0.0

    for i in range(num_samples):
        # Retrieve the top k predictions for the current sample
        pred_row = predictions[i][:k]
        target = targets[i]

        # Calculate Average Precision (AP) for this sample
        # Since there is only one correct label per image:
        # AP = 1/rank if the target is found within top k, else 0.
        # Rank is 1-based.
        ap = 0.0
        for rank, label in enumerate(pred_row):
            if label == target:
                ap = 1.0 / (rank + 1)
                break

        score_sum += ap

    return score_sum / num_samples
