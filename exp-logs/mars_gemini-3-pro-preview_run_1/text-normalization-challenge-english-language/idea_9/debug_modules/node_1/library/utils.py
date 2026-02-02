import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(
    class_counts: dict, vocab_size: int, smoothing: float = 0.5
) -> torch.Tensor:
    """
    Computes class weights for CrossEntropyLoss using smoothed inverse frequency.

    Formula: Weight_c = (Total_Samples / Count_c) ** smoothing

    This penalizes errors on rare classes without the instability of raw inverse frequency.

    Args:
        class_counts (dict): A dictionary mapping class indices (int) to their occurrence counts (int).
        vocab_size (int): The total number of unique classes (size of the output layer).
        smoothing (float): The exponent for smoothing. 0.5 corresponds to square root smoothing.

    Returns:
        torch.Tensor: A tensor of shape (vocab_size,) containing the computed weights.
    """
    # Initialize counts array
    counts = np.zeros(vocab_size, dtype=np.float32)

    # Populate counts from the dictionary
    for class_idx, count in class_counts.items():
        if 0 <= class_idx < vocab_size:
            counts[class_idx] = count

    # Calculate total samples (N)
    total_samples = np.sum(counts)

    # Handle zero counts to prevent division by zero.
    # If a class has 0 count, we treat it as having 1 count for weight calculation purposes.
    counts = np.maximum(counts, 1.0)

    # Compute weights: (N / N_c)^p
    weights = (total_samples / counts) ** smoothing

    return torch.tensor(weights, dtype=torch.float32)
