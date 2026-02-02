import torch
import os
import copy
from library.config import Config


def generate_soup_model(model, checkpoint_paths, device=None):
    """
    Generates a 'Model Soup' by averaging the weights of multiple checkpoints.

    This technique (Uniform Soup) averages the weights of models trained with
    different hyperparameters or at different epochs to find a flatter minimum
    in the loss landscape, often improving generalization and robustness.

    Args:
        model (torch.nn.Module): The base model architecture to load the averaged weights into.
        checkpoint_paths (list of str): List of file paths to the checkpoints to be averaged.
        device (str, optional): The device to perform the averaging on.
                                Defaults to CPU to save GPU memory.

    Returns:
        torch.nn.Module: The model with the averaged weights loaded.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths list cannot be empty.")

    if device is None:
        # Default to CPU for aggregation to avoid OOM on GPU with multiple state dicts
        device = "cpu"

    print(f"Generating Model Soup from {len(checkpoint_paths)} checkpoints...")

    # Initialize the averaged state dict
    avg_state_dict = None

    for i, path in enumerate(checkpoint_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        # Load checkpoint to the specific device (CPU recommended for summation)
        checkpoint = torch.load(path, map_location=device)

        # Extract the model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if avg_state_dict is None:
            # First checkpoint: initialize the accumulator
            # We use deepcopy to ensure we don't modify the loaded dict in place if cached
            avg_state_dict = copy.deepcopy(state_dict)

            # Ensure all tensors are float for averaging (handles integer buffers if any, though rare in weights)
            for key in avg_state_dict:
                if torch.is_floating_point(avg_state_dict[key]):
                    avg_state_dict[key] = avg_state_dict[key].clone()
        else:
            # Subsequent checkpoints: accumulate weights
            for key in avg_state_dict:
                if key in state_dict:
                    # Only average floating point tensors (weights/biases)
                    # Integer buffers (like num_batches_tracked) are usually kept from the first model
                    # or handled specifically, but simple summation works for weights.
                    if torch.is_floating_point(avg_state_dict[key]):
                        avg_state_dict[key] += state_dict[key]
                else:
                    raise KeyError(f"Key {key} missing in checkpoint {path}")

    # Compute the average
    num_models = len(checkpoint_paths)
    for key in avg_state_dict:
        if torch.is_floating_point(avg_state_dict[key]):
            avg_state_dict[key] = avg_state_dict[key] / num_models

    # Load the soup weights into the model
    # We move the model to the destination device (Config.DEVICE) if not already there
    # But here we just load the state dict. The model's current device is respected.
    model.load_state_dict(avg_state_dict)

    print("Model Soup generation complete.")
    return model
