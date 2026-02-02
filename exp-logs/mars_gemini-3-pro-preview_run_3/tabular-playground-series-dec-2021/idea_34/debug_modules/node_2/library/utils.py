import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CuDNN based on Config settings.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure CuDNN based on strategy
    if Config.DETERMINISTIC_CUDNN:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Performance mode: allow CuDNN to find best algorithms
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_submission(predictions, ids, output_path: str = None):
    """
    Formats and saves the final predictions to a CSV file.
    Handles input conversion from Tensor to Numpy if necessary.

    Args:
        predictions (array-like): Predicted class labels.
        ids (array-like): Corresponding IDs for the test set.
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_FILE.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_FILE

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert tensors to numpy if passed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()

    # Ensure inputs are 1D arrays
    predictions = np.array(predictions).flatten()
    ids = np.array(ids).flatten()

    # Create DataFrame matching competition format
    submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
