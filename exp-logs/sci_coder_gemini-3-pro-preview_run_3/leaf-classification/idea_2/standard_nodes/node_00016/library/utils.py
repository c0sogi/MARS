import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(predictions, test_ids, class_names, output_path):
    """
    Formats and saves the predictions to a CSV file in the required submission format.
    Applies probability clipping to avoid log loss extremes.

    Args:
        predictions (np.ndarray): Array of probability predictions (shape: [n_samples, n_classes]).
        test_ids (list or np.ndarray): List of test image IDs.
        class_names (list): List of class names corresponding to the columns of predictions.
        output_path (str): File path where the submission CSV will be saved.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Clip probabilities as per metric requirements: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    clipped_preds = np.clip(predictions, epsilon, 1 - epsilon)

    # Create DataFrame
    submission_df = pd.DataFrame(clipped_preds, columns=class_names)

    # Insert 'id' as the first column
    submission_df.insert(0, "id", test_ids)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
