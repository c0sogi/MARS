import os
import torch
from library.engine import evaluate, greedy_model_soup
from library.utils import get_logger

# Initialize logger for this module
logger = get_logger("soup")


def create_greedy_soup(model, loader, checkpoint_paths, device):
    """
    Iterates through saved checkpoints, evaluates them, and constructs a Greedy Model Soup.

    This function acts as the I/O driver for the soup algorithm:
    1. Loads state dictionaries from disk.
    2. Evaluates each to establish a baseline validation loss.
    3. Calls the engine's greedy_model_soup to perform the weight averaging optimization.

    Args:
        model (torch.nn.Module): The base model architecture instance.
        loader (torch.utils.data.DataLoader): Validation dataloader for evaluation.
        checkpoint_paths (list): List of file paths (strings) to .pth checkpoint files.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        dict: The state dictionary of the constructed greedy soup model.
    """
    logger.info(
        f"Starting Greedy Model Soup preparation with {len(checkpoint_paths)} checkpoints."
    )

    candidates = []

    # Ensure model is on the correct device for evaluation
    model.to(device)

    # 1. Load and Evaluate each checkpoint
    for i, path in enumerate(checkpoint_paths):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint path not found: {path}. Skipping.")
            continue

        logger.info(f"Processing candidate {i+1}/{len(checkpoint_paths)}: {path}")

        # Load state dict to CPU to avoid VRAM accumulation
        try:
            state_dict = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            continue

        # Load weights into model
        model.load_state_dict(state_dict)

        # Evaluate to get baseline loss
        # The evaluate function handles the forward pass and metric calculation
        loss = evaluate(model, loader, device)

        logger.info(f"Candidate {os.path.basename(path)} Loss: {loss}")

        # Append to candidates list formatted for the engine function
        candidates.append({"state_dict": state_dict, "loss": loss, "path": path})

    if not candidates:
        logger.error("No valid candidates found. Cannot construct soup.")
        return None

    # 2. Construct Soup using the engine's logic
    # library.engine.greedy_model_soup handles sorting and the greedy addition loop
    logger.info(f"Constructing soup from {len(candidates)} evaluated candidates...")

    final_soup_state = greedy_model_soup(model, loader, candidates, device)

    return final_soup_state
