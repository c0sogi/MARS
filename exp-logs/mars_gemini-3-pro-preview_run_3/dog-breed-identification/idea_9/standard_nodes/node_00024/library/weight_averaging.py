import os
import torch
from library.config import Config


def average_checkpoints(checkpoint_paths, output_path):
    """
    Loads state dictionaries from the provided checkpoint paths,
    computes the arithmetic mean of the weights (Model Soup),
    and saves the averaged model to the output path.

    Args:
        checkpoint_paths (list): List of file paths to the .pth checkpoints.
        output_path (str): Destination path to save the averaged state dictionary.

    Returns:
        dict: The averaged state dictionary.
    """
    if not checkpoint_paths:
        print("Error: No checkpoints provided for averaging.")
        return None

    num_models = len(checkpoint_paths)
    print(
        f"Starting Manual Weight Averaging (Model Soup) for {num_models} checkpoints..."
    )

    # Load the first checkpoint as the base accumulator
    # We use map_location='cpu' to avoid filling GPU memory with multiple models
    try:
        avg_state_dict = torch.load(checkpoint_paths[0], map_location="cpu")
    except Exception as e:
        print(f"Error loading base checkpoint {checkpoint_paths[0]}: {e}")
        return None

    # Iterate over the remaining checkpoints and sum their parameters
    for i, path in enumerate(checkpoint_paths[1:], start=1):
        try:
            state_dict = torch.load(path, map_location="cpu")

            for key in avg_state_dict:
                if key not in state_dict:
                    raise KeyError(f"Key '{key}' missing in checkpoint {path}")

                # In-place addition to save memory
                avg_state_dict[key] += state_dict[key]

        except Exception as e:
            print(f"Error processing checkpoint {path}: {e}")
            # In a strict pipeline, we might want to raise, but here we return None to signal failure
            return None

    # Compute the mean
    for key in avg_state_dict:
        tensor = avg_state_dict[key]

        if tensor.is_floating_point():
            # Standard floating point averaging
            avg_state_dict[key] = tensor / num_models
        else:
            # Handle integer tensors (e.g., BatchNorm num_batches_tracked)
            # Convert to float for division, then round/cast back to original dtype
            original_dtype = tensor.dtype
            avg_state_dict[key] = (tensor.float() / num_models).to(original_dtype)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    try:
        torch.save(avg_state_dict, output_path)
        print(f"Successfully saved averaged model to {output_path}")
    except Exception as e:
        print(f"Error saving averaged model to {output_path}: {e}")
        return None

    return avg_state_dict
