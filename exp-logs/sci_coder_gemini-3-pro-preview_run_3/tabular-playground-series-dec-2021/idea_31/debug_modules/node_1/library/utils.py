import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    According to project strategy (Lesson 00070), this explicitly disables
    strict CuDNN determinism to maximize kernel performance on the A100 GPU.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Performance Optimization: Disable strict determinism
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_submission(ids: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: Array of ID values.
        predictions: Array of predicted class labels.
        output_path: Path to save the submission CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame conforming to submission format
    submission = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    # Save to CSV without index
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
