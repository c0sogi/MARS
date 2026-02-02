import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(predictions, test_ids, class_names, output_path):
    """
    Formats predictions and saves them to a CSV file according to the competition format.

    Args:
        predictions (np.ndarray): Array of shape (n_samples, n_classes) containing probabilities.
        test_ids (list or np.ndarray): List of test image IDs.
        class_names (list): List of class names corresponding to the columns of predictions.
        output_path (str): File path where the submission CSV will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a DataFrame for the submission
    # The columns must be the breed names
    submission_df = pd.DataFrame(predictions, columns=class_names)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
