import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_submission(ids, predictions, class_names, output_path):
    """
    Formats predictions into a DataFrame and saves it as a CSV file.

    Args:
        ids (array-like): List or array of image IDs.
        predictions (array-like): Matrix of predicted probabilities (shape: n_samples x n_classes).
        class_names (list): List of class names corresponding to the columns of predictions.
        output_path (str): The file path where the submission CSV should be saved.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=class_names)
    submission_df.insert(0, "id", ids)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
