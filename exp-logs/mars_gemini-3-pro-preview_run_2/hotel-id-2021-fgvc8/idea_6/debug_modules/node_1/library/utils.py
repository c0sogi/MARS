import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures CuDNN to be deterministic.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_average_precision(predictions, targets, k: int = 5) -> float:
    """
    Calculates the Mean Average Precision @ k (MAP@k) for single-label classification.

    In this task, there is exactly one relevant item (the true hotel_id) per query.
    Therefore, AP@k for a single query is (1/rank) if the true label is at 'rank'
    (1-indexed) within the top k predictions, and 0 otherwise.

    Args:
        predictions (list, np.ndarray, torch.Tensor): A collection of predicted labels.
            Shape should be (N, M) where M >= k.
            Ordered by confidence (highest first).
        targets (list, np.ndarray, torch.Tensor): A collection of ground truth labels.
            Shape should be (N,).
        k (int): The cutoff rank for the metric. Defaults to 5.

    Returns:
        float: The MAP@k score.
    """
    # Convert PyTorch tensors to NumPy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Convert NumPy arrays to lists for consistent and safe indexing
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    score_sum = 0.0
    n_samples = len(targets)

    if n_samples == 0:
        return 0.0

    for preds, target in zip(predictions, targets):
        # Consider only the top k predictions
        top_k_preds = preds[:k]

        if target in top_k_preds:
            # rank is 0-indexed in list, so we add 1 for 1-based ranking
            rank = top_k_preds.index(target) + 1
            score_sum += 1.0 / rank

    return score_sum / n_samples
