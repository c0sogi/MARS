import torch
from library.config import seed_everything


def get_device() -> torch.device:
    """
    Determines the computational device to use.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
