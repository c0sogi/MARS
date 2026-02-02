import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    """
    Recursively unwraps a model from containers like AveragedModel or DataParallel
    to retrieve the underlying base model.
    """
    # Check for SWA AveragedModel or DataParallel/DistributedDataParallel
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        return unwrap_model(model.module)
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def save_clean_checkpoint(model, path: str):
    """
    Saves the model state dictionary after unwrapping it from any wrappers
    (e.g., SWA AveragedModel, DataParallel). This ensures the checkpoint
    keys match the original model architecture for inference.

    Args:
        model: The PyTorch model (or wrapped model) to save.
        path (str): The file path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Unwrap the model to get the raw architecture
    clean_model = unwrap_model(model)

    # Save the state dictionary
    torch.save(clean_model.state_dict(), path)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
