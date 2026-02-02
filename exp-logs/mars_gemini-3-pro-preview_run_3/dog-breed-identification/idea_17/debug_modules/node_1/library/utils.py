import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_model_soup(model_paths: list, save_path: str):
    """
    Loads state dictionaries from the provided model paths, computes the
    arithmetic mean of the weights, and saves the fused model to disk.

    This implements the 'Model Soup' strategy (Greedy/Uniform) manually,
    allowing for weight-space ensembling of fine-tuned models.

    Args:
        model_paths (list): List of file paths to .pth model checkpoints.
        save_path (str): Destination path to save the soup model.
    """
    if not model_paths:
        print("No model paths provided for soup generation.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Generating model soup from {len(model_paths)} models...")

    # Load the first model to serve as the base accumulator
    # We map to CPU to save GPU memory during aggregation
    try:
        base_chk = torch.load(model_paths[0], map_location="cpu")
    except Exception as e:
        print(f"Error loading {model_paths[0]}: {e}")
        return

    # Handle case where checkpoint is a dict containing 'model_state_dict' or similar
    if isinstance(base_chk, dict) and "model_state_dict" in base_chk:
        base_state_dict = base_chk["model_state_dict"]
    elif isinstance(base_chk, dict) and "state_dict" in base_chk:
        base_state_dict = base_chk["state_dict"]
    else:
        base_state_dict = base_chk

    # Iterate over the remaining models
    for i in range(1, len(model_paths)):
        path = model_paths[i]
        try:
            current_chk = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Unwrap if necessary
        if isinstance(current_chk, dict) and "model_state_dict" in current_chk:
            current_state_dict = current_chk["model_state_dict"]
        elif isinstance(current_chk, dict) and "state_dict" in current_chk:
            current_state_dict = current_chk["state_dict"]
        else:
            current_state_dict = current_chk

        # Add weights to base
        for key in base_state_dict:
            if key in current_state_dict:
                # We perform in-place addition
                # Ensure types match (e.g. if one is float and other is half, though unlikely here)
                base_state_dict[key] += current_state_dict[key]
            else:
                # This might happen if architectures differ slightly or strict loading wasn't used
                pass

    # Compute mean
    n = len(model_paths)
    for key in base_state_dict:
        # Check if the tensor is a floating point type before division
        # Integer tensors (like batches_tracked) are usually summed, but for soup we average.
        if torch.is_floating_point(base_state_dict[key]):
            base_state_dict[key] = base_state_dict[key] / n
        else:
            # For integer types, we use floor division to maintain type
            base_state_dict[key] = base_state_dict[key] // n

    # Save the soup
    torch.save(base_state_dict, save_path)
    print(f"Model soup saved to {save_path}")
