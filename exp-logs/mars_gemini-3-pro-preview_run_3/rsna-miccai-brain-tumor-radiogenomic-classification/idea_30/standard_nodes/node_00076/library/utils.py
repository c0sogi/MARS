import torch
from library.config import set_seed, SEED, DEVICE


def seed_everything(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function wraps the set_seed implementation provided in the configuration
    library to ensure consistent seeding behavior across the pipeline.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    set_seed(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for training and inference.

    Uses the DEVICE configuration determined in library.config (cuda if available, else cpu).

    Returns:
        torch.device: The PyTorch device object.
    """
    return torch.device(DEVICE)
