import torch
from library.config import seed_everything, DEVICE


def get_device(device_name=None):
    """
    Returns the PyTorch device to be used for computation.

    Args:
        device_name (str, optional): The name of the device (e.g., 'cpu', 'cuda').
                                     If None, defaults to the DEVICE constant defined
                                     in library.config (which auto-detects GPU availability).

    Returns:
        torch.device: The configured PyTorch device.
    """
    if device_name is not None:
        return torch.device(device_name)
    return torch.device(DEVICE)
