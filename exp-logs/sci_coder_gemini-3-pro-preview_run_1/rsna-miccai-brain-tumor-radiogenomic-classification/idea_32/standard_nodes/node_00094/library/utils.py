import torch
from library.config import seed_everything, DEVICE


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for computation.

    This function utilizes the global DEVICE configuration determined in
    library.config (which checks for CUDA availability) and returns
    a torch.device object.

    Returns:
        torch.device: The device object ('cuda' or 'cpu').
    """
    return torch.device(DEVICE)


# Note: seed_everything is imported from library.config and is available
# in this module's namespace, fulfilling the requirement to include it
# without re-implementation.
