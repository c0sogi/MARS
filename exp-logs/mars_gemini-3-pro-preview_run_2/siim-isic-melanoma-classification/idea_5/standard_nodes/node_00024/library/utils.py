import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Any


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def average_checkpoints(checkpoint_paths: List[str]) -> Dict[str, Any]:
    """
    Loads multiple model checkpoints and averages their state dictionaries parameter-wise.
    Handles type conversion to ensure integer parameters (like BatchNorm counters) are preserved correctly.

    Args:
        checkpoint_paths (List[str]): List of file paths to the .pth checkpoint files.

    Returns:
        Dict[str, Any]: The averaged state dictionary ready to be loaded into the model.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for averaging.")

    # Load the first checkpoint to serve as the reference for keys and types
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle case where checkpoint is a dict containing 'model_state_dict'
    if "model_state_dict" in first_ckpt:
        ref_state_dict = first_ckpt["model_state_dict"]
    else:
        ref_state_dict = first_ckpt

    # Initialize the accumulator with the first checkpoint's values converted to float
    # We use float for accumulation to avoid overflow and allow precise division
    avg_state_dict = {k: v.clone().float() for k, v in ref_state_dict.items()}

    # Iterate over the remaining checkpoints
    for path in checkpoint_paths[1:]:
        ckpt = torch.load(path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        for k, v in state_dict.items():
            if k in avg_state_dict:
                avg_state_dict[k] += v.float()

    # Compute average and cast back to original types
    num_ckpts = len(checkpoint_paths)
    final_state_dict = {}

    for k, v in avg_state_dict.items():
        # Divide by number of checkpoints
        avg_val = v / num_ckpts

        # Restore original data type
        ref_val = ref_state_dict[k]
        if ref_val.is_floating_point():
            final_state_dict[k] = avg_val.to(ref_val.dtype)
        else:
            # For integer types (e.g., LongTensor for num_batches_tracked), round to nearest integer
            final_state_dict[k] = torch.round(avg_val).to(ref_val.dtype)

    return final_state_dict
