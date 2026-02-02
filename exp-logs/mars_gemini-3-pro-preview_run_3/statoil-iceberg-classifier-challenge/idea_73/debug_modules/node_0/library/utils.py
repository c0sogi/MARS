import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_mad(x, dim=None, keepdim=False):
    """
    Calculates the Mean Absolute Deviation (MAD) of a tensor.
    MAD is defined as Mean(|x - Mean(x)|).

    This function is useful for texture analysis in signal processing,
    providing a dispersion metric robust to outliers.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.

    Returns:
        torch.Tensor: The MAD of the input tensor.
    """
    # Calculate the mean along the specified dimension(s)
    if dim is not None:
        mean = x.mean(dim=dim, keepdim=True)
    else:
        mean = x.mean()

    # Calculate the absolute deviation from the mean
    deviation = torch.abs(x - mean)

    # Calculate the mean of these deviations
    if dim is not None:
        mad = deviation.mean(dim=dim, keepdim=keepdim)
    else:
        mad = deviation.mean()

    return mad


def impute_inc_angles(train_angles, *other_angles_sets):
    """
    Imputes missing incidence angles using the median of the training set.

    To prevent data leakage, the median is calculated strictly from the
    training data (ignoring NaNs) and then applied to fill NaNs in both
    the training set and any other provided sets (e.g., validation, test).

    Args:
        train_angles (np.ndarray): Array of incidence angles for training data.
                                   May contain NaNs.
        *other_angles_sets (np.ndarray): Variable number of other angle arrays
                                         (e.g., validation set, test set) to
                                         impute using the training median.

    Returns:
        tuple or np.ndarray: If only train_angles is provided, returns the imputed
                             train array. If other sets are provided, returns a
                             tuple of (imputed_train, imputed_other_1, ...).
    """
    # Calculate median from training data, ignoring NaNs
    median_val = np.nanmedian(train_angles)

    # Impute training data
    # We use np.where to replace NaNs with the calculated median
    train_imputed = np.where(np.isnan(train_angles), median_val, train_angles)

    results = [train_imputed]

    # Apply the same median to other datasets
    for angles in other_angles_sets:
        imputed = np.where(np.isnan(angles), median_val, angles)
        results.append(imputed)

    if len(results) == 1:
        return results[0]
    return tuple(results)
