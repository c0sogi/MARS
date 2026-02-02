import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_weights(state_dict1, state_dict2, alpha=0.5):
    """
    Mathematically averages the state dictionaries of two models.
    Used for the Greedy Model Soup strategy.

    Formula: new_weight = alpha * weight1 + (1 - alpha) * weight2

    Args:
        state_dict1 (dict): State dictionary of the first model.
        state_dict2 (dict): State dictionary of the second model.
        alpha (float): Weighting factor for the first model. Defaults to 0.5.

    Returns:
        dict: A new state dictionary containing the averaged weights.
    """
    new_state_dict = {}

    for key in state_dict1.keys():
        if key in state_dict2:
            param1 = state_dict1[key]
            param2 = state_dict2[key]

            # Perform weighted averaging
            # We assume tensors are on compatible devices or CPU.
            # Operations on tensors preserve the device of the inputs.
            averaged_param = alpha * param1 + (1 - alpha) * param2

            # Handle non-floating point tensors (e.g., BatchNorm num_batches_tracked)
            # These are typically LongTensors and must remain so.
            if not param1.is_floating_point():
                averaged_param = averaged_param.round().to(param1.dtype)

            new_state_dict[key] = averaged_param
        else:
            # If key is missing in state_dict2, retain state_dict1's value (should not happen in identical archs)
            new_state_dict[key] = state_dict1[key].clone()

    return new_state_dict


def calculate_metric(y_true, y_pred):
    """
    Computes the Multi Class Log Loss.

    Args:
        y_true (array-like): Ground truth labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles multiclass targets automatically
    # It also handles clipping probabilities to avoid log(0)
    return log_loss(y_true, y_pred)
