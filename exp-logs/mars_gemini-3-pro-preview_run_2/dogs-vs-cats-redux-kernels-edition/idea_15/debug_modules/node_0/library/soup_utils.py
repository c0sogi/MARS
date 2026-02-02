import os
import torch


def average_weights(checkpoint_paths):
    """
    Computes the Model Soup (weight average) of the provided checkpoints.

    This function loads multiple PyTorch model checkpoints, averages their parameters
    element-wise, and returns a single state dictionary. It handles type conversion
    (accumulating as float) and restores original data types (e.g., rounding for
    integer buffers like BatchNorm counters).

    Args:
        checkpoint_paths (list[str]): List of file paths to the model checkpoints (.pth files).

    Returns:
        dict: A state dictionary containing the averaged weights.

    Raises:
        ValueError: If the checkpoint_paths list is empty.
        FileNotFoundError: If a checkpoint file does not exist.
        KeyError: If checkpoints have mismatching keys (different architectures).
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging.")

    # Load the first checkpoint to establish the structure and initial sum
    first_path = checkpoint_paths[0]
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Checkpoint not found: {first_path}")

    # Load to CPU to save GPU memory during the averaging process
    loaded_data = torch.load(first_path, map_location="cpu")

    # Extract state_dict if the checkpoint is wrapped in a dictionary
    # Common keys are 'model' or 'state_dict'
    if "model" in loaded_data and isinstance(loaded_data["model"], dict):
        base_state_dict = loaded_data["model"]
    elif "state_dict" in loaded_data and isinstance(loaded_data["state_dict"], dict):
        base_state_dict = loaded_data["state_dict"]
    else:
        base_state_dict = loaded_data

    # Initialize the sum dictionary with float values for precision
    sum_state_dict = {}
    for key, value in base_state_dict.items():
        sum_state_dict[key] = value.clone().float()

    num_models = len(checkpoint_paths)

    # Iterate through the remaining checkpoints
    for i in range(1, num_models):
        path = checkpoint_paths[i]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data = torch.load(path, map_location="cpu")

        # Extract state_dict
        if "model" in data and isinstance(data["model"], dict):
            current_state_dict = data["model"]
        elif "state_dict" in data and isinstance(data["state_dict"], dict):
            current_state_dict = data["state_dict"]
        else:
            current_state_dict = data

        # Accumulate weights
        for key in sum_state_dict:
            if key not in current_state_dict:
                raise KeyError(f"Key '{key}' missing in checkpoint: {path}")

            # Add to accumulator
            sum_state_dict[key] += current_state_dict[key].float()

    # Compute average and restore original data types
    avg_state_dict = {}
    for key, value in sum_state_dict.items():
        # Compute mean
        avg_val = value / num_models

        # Restore original dtype from the base model
        original_tensor = base_state_dict[key]

        if not torch.is_floating_point(original_tensor):
            # For integer buffers (e.g., BatchNorm num_batches_tracked), round to nearest integer
            avg_val = torch.round(avg_val)

        avg_state_dict[key] = avg_val.to(original_tensor.dtype)

    return avg_state_dict
