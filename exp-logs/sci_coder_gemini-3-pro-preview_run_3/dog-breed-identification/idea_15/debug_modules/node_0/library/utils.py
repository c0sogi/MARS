import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=None):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int, optional): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Multi Class Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (n_samples, ) or one-hot encoded.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    return log_loss(y_true, y_pred)


def average_weights(checkpoint_paths, output_path):
    """
    Averages the state dictionaries of multiple model checkpoints (Manual Soup).

    This function loads checkpoints one by one to save memory, accumulates the
    parameter values, divides by the number of checkpoints, and saves the
    averaged state dictionary to the output path.

    Args:
        checkpoint_paths (list): List of strings, paths to the .pth model files.
        output_path (str): Path where the averaged model should be saved.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths list is empty. Cannot average weights.")

    # Load the first checkpoint to initialize the accumulator
    # Map to CPU to avoid GPU OOM during aggregation
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle cases where the checkpoint is a dict containing 'model_state_dict'
    # or just the state dictionary itself.
    if isinstance(first_ckpt, dict) and "model_state_dict" in first_ckpt:
        soup_state_dict = first_ckpt["model_state_dict"]
    else:
        soup_state_dict = first_ckpt

    # Convert parameters to float for precise averaging
    for key in soup_state_dict:
        soup_state_dict[key] = soup_state_dict[key].float()

    num_models = len(checkpoint_paths)

    # Iterate through the rest of the checkpoints
    for i in range(1, num_models):
        current_path = checkpoint_paths[i]
        ckpt = torch.load(current_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            current_state_dict = ckpt["model_state_dict"]
        else:
            current_state_dict = ckpt

        # Accumulate weights
        for key in soup_state_dict:
            # Ensure keys match; strictly they should for identical architectures
            if key in current_state_dict:
                soup_state_dict[key] += current_state_dict[key].float()
            else:
                raise KeyError(f"Key {key} missing in checkpoint {current_path}")

    # Compute the average
    for key in soup_state_dict:
        soup_state_dict[key] /= num_models

    # Save the averaged weights
    # We save as a simple state dict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(soup_state_dict, output_path)

    print(f"Successfully averaged weights from {num_models} models.")
    print(f"Saved averaged model to: {output_path}")
