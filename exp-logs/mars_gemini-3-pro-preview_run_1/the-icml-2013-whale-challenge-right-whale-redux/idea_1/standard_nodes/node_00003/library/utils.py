import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_submission(predictions, test_ids, output_path: str = Config.SUBMISSION_PATH):
    """
    Saves the model predictions to a CSV file in the format required for submission.

    Args:
        predictions (array-like): List or array of predicted probabilities.
        test_ids (array-like): List or array of clip filenames corresponding to predictions.
        output_path (str): The file path where the submission CSV will be saved.
                           Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({"clip": test_ids, "probability": predictions})

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
