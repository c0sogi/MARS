import os
import copy
import torch
import numpy as np
from library.config import Config
from library.model import DogBreedModel
from library.engine import validate


def get_averaged_state_dict(state_dicts):
    """
    Computes the arithmetic mean of a list of state dictionaries.

    Args:
        state_dicts (list): List of state dictionaries (ordered dicts).

    Returns:
        dict: Averaged state dictionary.
    """
    if not state_dicts:
        return None

    # Use the first state dict as a template
    # We perform calculations on CPU to save GPU memory
    avg_state = {}

    # Get keys from the first model
    keys = state_dicts[0].keys()

    # Number of models to average
    n = len(state_dicts)

    for key in keys:
        # Start with the tensor from the first model
        # Clone to avoid modifying the original
        key_sum = state_dicts[0][key].clone().to("cpu")

        # Add tensors from the rest of the models
        for i in range(1, n):
            key_sum += state_dicts[i][key].to("cpu")

        # Divide by n to get the average
        avg_state[key] = key_sum / n

    return avg_state


def create_greedy_soup(checkpoint_paths, val_loader, device):
    """
    Implements the Greedy Model Soup algorithm.

    1. Evaluates all checkpoints on the validation set.
    2. Sorts them by Log Loss.
    3. Iteratively adds models to the soup if they improve performance.

    Args:
        checkpoint_paths (list): List of file paths to model checkpoints (.pth).
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: The state dictionary of the best greedy soup model.
    """
    print(
        f"Starting Greedy Soup construction with {len(checkpoint_paths)} checkpoints..."
    )

    # Initialize the model structure
    # pretrained=False because we will load weights from checkpoints
    model = DogBreedModel(pretrained=False)
    model.to(device)

    # Store loaded state dicts and metrics
    # Format: {'path': str, 'state_dict': dict, 'loss': float}
    candidates = []

    # ---------------------------------------------------------
    # Step 1: Evaluate individual models
    # ---------------------------------------------------------
    print("Step 1: Evaluating individual models...")
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"  Warning: Checkpoint not found at {path}, skipping.")
            continue

        # Load state dict to CPU to preserve GPU memory
        state_dict = torch.load(path, map_location="cpu")

        # Load into model for validation
        # load_state_dict handles moving CPU tensors to the GPU model
        model.load_state_dict(state_dict)

        # Validate
        loss, _, _ = validate(model, val_loader, device)
        print(f"  Model {os.path.basename(path)}: Log Loss = {loss}")

        candidates.append({"path": path, "state_dict": state_dict, "loss": loss})

    if not candidates:
        print("Error: No valid checkpoints evaluated.")
        return None

    # ---------------------------------------------------------
    # Step 2: Sort candidates by Loss (Ascending)
    # ---------------------------------------------------------
    candidates.sort(key=lambda x: x["loss"])

    # ---------------------------------------------------------
    # Step 3: Greedy Construction
    # ---------------------------------------------------------
    print("\nStep 2: Constructing Greedy Soup...")

    # Start with the best single model
    soup_ingredients = [candidates[0]]
    best_loss = candidates[0]["loss"]

    print(
        f"  Baseline (Best Single): {os.path.basename(candidates[0]['path'])} | Loss: {best_loss}"
    )

    # Iteratively try adding other models
    for i in range(1, len(candidates)):
        candidate = candidates[i]
        candidate_name = os.path.basename(candidate["path"])

        # Create a temporary list of ingredients including the new candidate
        temp_ingredients = soup_ingredients + [candidate]

        # Average weights of all ingredients
        temp_state_dicts = [c["state_dict"] for c in temp_ingredients]
        avg_state_dict = get_averaged_state_dict(temp_state_dicts)

        # Load averaged weights and validate
        model.load_state_dict(avg_state_dict)
        loss, _, _ = validate(model, val_loader, device)

        print(f"  Trying + {candidate_name}... Combined Loss: {loss}")

        # Greedy acceptance criterion
        if loss < best_loss:
            diff = best_loss - loss
            print(f"    -> ACCEPTED (Improved by {diff})")
            best_loss = loss
            soup_ingredients.append(candidate)
        else:
            print(f"    -> REJECTED (No improvement)")

    # ---------------------------------------------------------
    # Step 4: Finalize
    # ---------------------------------------------------------
    print("-" * 30)
    print(f"Greedy Soup Complete.")
    print(f"Ingredients: {len(soup_ingredients)} models.")
    print(f"Final Best Log Loss: {best_loss}")

    # Compute final soup weights
    final_state_dicts = [c["state_dict"] for c in soup_ingredients]
    final_soup_dict = get_averaged_state_dict(final_state_dicts)

    return final_soup_dict
