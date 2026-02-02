import torch
from library.config import Config


def seed_everything(seed: int = None) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    # Delegate to the centralized Config method to avoid re-implementation
    # and ensure consistent seeding logic across the pipeline.
    Config.set_seed(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for computation.

    Returns:
        torch.device: The device object (CPU or CUDA) defined in Config.
    """
    return torch.device(Config.DEVICE)
