import os
import random
import numpy as np
import torch
import library.config as config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    # This might impact performance slightly but guarantees reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map5(predictions, targets):
    """
    Calculates Mean Average Precision @ 5 (MAP@5).

    For a single-label classification task, MAP@5 is the mean of (1 / rank)
    where rank is the 1-based position of the true label in the top-5 predictions.
    If the true label is not in the top-5, the score is 0.

    Args:
        predictions (list of list of str): A list where each element is a list of
                                           predicted class labels (strings), ordered by confidence.
                                           Only the first 5 predictions are considered.
        targets (list of str): A list of ground truth class labels (strings).

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    score_sum = 0.0
    n = len(targets)

    if n == 0:
        return 0.0

    for preds, target in zip(predictions, targets):
        # Ensure we only look at the top 5 predictions
        preds_top5 = preds[:5]

        if target in preds_top5:
            # index() returns 0-based index, so add 1 for rank
            rank = preds_top5.index(target) + 1
            score_sum += 1.0 / rank
        else:
            score_sum += 0.0

    return score_sum / n
