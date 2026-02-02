import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def save_submission(
    predictions, test_ids, class_names, output_path=Config.SUBMISSION_PATH
):
    """
    Formats and saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): A numpy array of shape (n_samples, n_classes) containing probabilities.
        test_ids (list or np.ndarray): A list or array of image IDs corresponding to the predictions.
        class_names (list): A list of class names (strings) corresponding to the columns of predictions.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Ensure predictions are within the safe range for log-loss metric
    # As per metric description: max(min(p, 1-10^-15), 10^-15)
    epsilon = Config.PROB_EPSILON
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

    # Create DataFrame
    # Note: The order of columns in predictions must match the order in class_names
    submission_df = pd.DataFrame(predictions, columns=class_names)

    # Insert 'id' column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
