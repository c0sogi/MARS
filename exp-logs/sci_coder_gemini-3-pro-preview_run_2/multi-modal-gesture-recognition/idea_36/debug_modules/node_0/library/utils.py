import os
import random
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    # This might impact performance but guarantees reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_padding_mask(lengths, max_len=None):
    """
    Generates a boolean mask for variable length sequences.

    Args:
        lengths (torch.Tensor): Tensor of shape (batch_size,) containing sequence lengths.
        max_len (int, optional): The maximum sequence length. If None, it is inferred from the max value in lengths.

    Returns:
        torch.Tensor: Boolean tensor of shape (batch_size, max_len).
                      True indicates a valid position (actual data), False indicates padding.
    """
    if max_len is None:
        max_len = lengths.max().item()

    # Create a range tensor [0, 1, ..., max_len-1] on the same device as lengths
    # Shape: (1, max_len)
    indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)

    # Expand lengths to (batch_size, 1) to allow broadcasting
    lengths_expanded = lengths.unsqueeze(1)

    # Create mask: indices < lengths
    # Shape: (batch_size, max_len)
    # Positions where the index is less than the sequence length are valid (True)
    mask = indices < lengths_expanded

    return mask
