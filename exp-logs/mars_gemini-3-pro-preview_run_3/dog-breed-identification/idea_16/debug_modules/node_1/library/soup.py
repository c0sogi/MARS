import os
import torch
import copy
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("soup", os.path.join(Config.working_dir, "soup.log"))


def create_model_soup(checkpoint_paths, save_path):
    """
    Averages the weights of multiple model checkpoints to create a 'Model Soup'.

    Args:
        checkpoint_paths (list): List of file paths to the model checkpoints (state_dicts).
        save_path (str): Path where the aggregated soup model will be saved.
    """
    num_checkpoints = len(checkpoint_paths)
    if num_checkpoints == 0:
        logger.warning("No checkpoints provided for model soup generation.")
        return

    logger.info(f"Starting Model Soup generation with {num_checkpoints} checkpoints.")

    # Load the first checkpoint to initialize the soup dictionary
    first_path = checkpoint_paths[0]
    logger.info(f"Loading base checkpoint: {first_path}")

    # Load to CPU to avoid OOM errors during aggregation
    base_state = torch.load(first_path, map_location="cpu")

    # Handle cases where the checkpoint is a dict containing the state_dict
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
    elif "state_dict" in base_state:
        base_state = base_state["state_dict"]
    elif "model" in base_state:
        base_state = base_state["model"]

    # Create a deep copy to serve as the accumulator
    soup_state_dict = copy.deepcopy(base_state)

    # Iterate over the remaining checkpoints and accumulate weights
    for i in range(1, num_checkpoints):
        path = checkpoint_paths[i]
        logger.info(f"Aggregating checkpoint {i+1}/{num_checkpoints}: {path}")

        curr_state = torch.load(path, map_location="cpu")

        # Unwrap state dict if necessary
        if "model_state_dict" in curr_state:
            curr_state = curr_state["model_state_dict"]
        elif "state_dict" in curr_state:
            curr_state = curr_state["state_dict"]
        elif "model" in curr_state:
            curr_state = curr_state["model"]

        # Sum parameters
        for key in soup_state_dict:
            if key in curr_state:
                # Ensure tensors are on the same device and type before adding
                soup_state_dict[key] += curr_state[key]
            else:
                logger.warning(
                    f"Key {key} missing in checkpoint {path}. Skipping parameter."
                )

    # Average the accumulated weights
    logger.info("Averaging weights...")
    for key in soup_state_dict:
        # In-place division
        if soup_state_dict[key].is_floating_point():
            soup_state_dict[key].div_(num_checkpoints)
        else:
            # For non-floating point (e.g. num_batches_tracked), we usually keep the first or average integer
            # Standard practice for souping is often just averaging floats.
            # Long tensors (like buffers) might need float conversion or just floor division.
            # Here we assume float division is appropriate for weights/biases.
            soup_state_dict[key] = soup_state_dict[key] // num_checkpoints

    # Save the resulting soup model
    logger.info(f"Saving Model Soup to {save_path}")

    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(soup_state_dict, save_path)
    logger.info("Model Soup generation complete.")
