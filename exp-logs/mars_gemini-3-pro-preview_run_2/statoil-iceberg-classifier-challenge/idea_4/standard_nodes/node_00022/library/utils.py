import os
import random
import copy
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_best_model(model, path):
    """
    Saves the model state dictionary to the specified path.
    Crucially uses copy.deepcopy to safely store the model state dictionary,
    preventing the 'mutable reference' bug where the saved model inadvertently
    updates during subsequent training steps.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The file path where the model checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Create a deep copy of the state dictionary
    # This ensures we hold a snapshot of the weights at this exact moment,
    # independent of any future updates to the model object in memory.
    best_model_wts = copy.deepcopy(model.state_dict())

    # Save the deep-copied state dictionary to disk
    torch.save(best_model_wts, path)
