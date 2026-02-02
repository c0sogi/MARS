import os
import re
import json
import pickle
import torch
from library.config import Config, set_seed


def is_semiotic(text):
    """
    Checks if the input text contains any digits.
    Used to identify tokens that likely require normalization (e.g., numbers, dates).

    Args:
        text (str): The token text to check.

    Returns:
        bool: True if the text contains a digit, False otherwise.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\d", text))


def save_checkpoint(state, filename):
    """
    Saves a PyTorch model checkpoint to the specified filename.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a PyTorch model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Handle cases where the state_dict might be nested under 'state_dict' or 'model_state_dict'
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_pickle(obj, filename):
    """
    Saves an object to a file using pickle.

    Args:
        obj (Any): The object to save.
        filename (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    """
    Loads an object from a pickle file.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        Any: The loaded object.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Pickle file not found: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_json(obj, filename):
    """
    Saves an object to a JSON file.

    Args:
        obj (Any): The object to save (must be JSON serializable).
        filename (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(filename):
    """
    Loads an object from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        Any: The loaded object.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
