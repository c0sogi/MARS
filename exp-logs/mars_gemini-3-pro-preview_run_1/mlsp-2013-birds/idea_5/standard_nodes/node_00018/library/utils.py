import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary to save (e.g., model weights, optimizer, epoch).
        is_best (bool): If True, creates a copy of this checkpoint as the 'best' model.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)

    if is_best:
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        # Create a 'best' version of the filename in the same directory
        best_filename = os.path.join(dirname, f"best_{basename}")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads the model (and optionally optimizer) state from a checkpoint file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary loaded from the file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load on the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Support both full checkpoint dicts and direct state_dicts
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
    else:
        # Assume the file is just the state dict
        model.load_state_dict(checkpoint)
        return checkpoint


def create_submission(predictions, test_ids, submission_path):
    """
    Generates the submission CSV file in the required competition format.

    Format:
        Id,Probability
        rec_id*100+species_id, probability

    Args:
        predictions (np.ndarray): Shape (N_samples, 19) with probability scores.
        test_ids (list or np.ndarray): Shape (N_samples,) with rec_ids.
        submission_path (str): Path to save the CSV file.
    """
    if predictions.shape[0] != len(test_ids):
        raise ValueError(
            f"Mismatch between predictions count ({predictions.shape[0]}) and test_ids count ({len(test_ids)})"
        )

    if predictions.shape[1] != Config.NUM_CLASSES:
        raise ValueError(
            f"Predictions must have {Config.NUM_CLASSES} columns, got {predictions.shape[1]}"
        )

    records = []
    for i, rec_id in enumerate(test_ids):
        for species_idx in range(Config.NUM_CLASSES):
            # Construct the composite ID as per task description
            composite_id = int(rec_id * 100 + species_idx)
            prob = float(predictions[i, species_idx])
            records.append({"Id": composite_id, "Probability": prob})

    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save to CSV
    df.to_csv(submission_path, index=False)
