import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED, DEVICE


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, path):
    """
    Saves the PyTorch model state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the model state dict will be saved.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=DEVICE):
    """
    Loads the PyTorch model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the saved model state dict.
        device (str): The device to map the location to (default: from config).

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_scalers(scalers, path):
    """
    Saves the scalers (dictionary of tensors) to disk using torch.save.

    Args:
        scalers (dict): Dictionary containing scaler parameters (e.g., mean, std).
        path (str): The file path to save the scalers.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    torch.save(scalers, path)


def load_scalers(path):
    """
    Loads the scalers from disk.

    Args:
        path (str): The file path to load the scalers from.

    Returns:
        dict: The loaded scaler parameters.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scalers file not found: {path}")
    return torch.load(path)


def read_metadata(path):
    """
    Reads a CSV metadata file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def save_submission(ids, formation_preds, bandgap_preds, path):
    """
    Saves the predictions to a CSV file in the required submission format.

    Args:
        ids (list or np.array): List of material IDs.
        formation_preds (list or np.array): Predicted formation energies.
        bandgap_preds (list or np.array): Predicted bandgap energies.
        path (str): Path to save the submission CSV.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_preds,
            "bandgap_energy_ev": bandgap_preds,
        }
    )
    df.to_csv(path, index=False)
