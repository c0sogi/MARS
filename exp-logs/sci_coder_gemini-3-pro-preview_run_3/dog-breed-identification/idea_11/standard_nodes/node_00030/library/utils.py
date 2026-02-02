import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode ensures reproducibility but may impact performance slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model_soup(checkpoint_paths, device=Config.DEVICE):
    """
    Averages the weights of multiple model checkpoints to create a 'Model Soup'.

    This function loads multiple state dictionaries, averages their weights key-by-key,
    and returns a single consolidated state dictionary. This corresponds to the
    Level 1 (Intra-Fold) ensemble strategy.

    Args:
        checkpoint_paths (list): List of file paths to the model checkpoints (.pth files).
        device (str): Device to load the checkpoints onto (default: Config.DEVICE).

    Returns:
        dict: A state dictionary containing the averaged weights.
    """
    if not checkpoint_paths:
        print("No checkpoints provided for Model Soup.")
        return None

    print(f"Creating Model Soup from {len(checkpoint_paths)} checkpoints...")

    # Load the first checkpoint to initialize the soup
    # We use map_location to control where the tensors are loaded (e.g., 'cpu' to save VRAM)
    first_ckpt_path = checkpoint_paths[0]
    soup_state_dict = torch.load(first_ckpt_path, map_location=device)

    # If only one checkpoint is provided, return it directly
    if len(checkpoint_paths) == 1:
        return soup_state_dict

    # Accumulate weights from the remaining checkpoints
    for path in checkpoint_paths[1:]:
        current_state_dict = torch.load(path, map_location=device)

        for key in soup_state_dict:
            if key in current_state_dict:
                # Accumulate values. Note: This creates new tensors.
                soup_state_dict[key] = soup_state_dict[key] + current_state_dict[key]
            else:
                # In a consistent training pipeline, keys should always match.
                pass

    # Average the accumulated weights
    num_models = len(checkpoint_paths)
    for key in soup_state_dict:
        # Handle floating point tensors (weights, biases, running_mean, etc.)
        if torch.is_floating_point(soup_state_dict[key]):
            soup_state_dict[key] = soup_state_dict[key] / num_models
        else:
            # Handle integer tensors (e.g., num_batches_tracked in BatchNorm)
            # Convert to float for division to get the mean, then cast back to original type.
            # This ensures the state_dict remains compatible with the model definition.
            soup_state_dict[key] = (soup_state_dict[key].float() / num_models).to(
                soup_state_dict[key].dtype
            )

    print("Model Soup creation complete.")
    return soup_state_dict
