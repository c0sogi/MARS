import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(
    ids, probabilities, class_names, output_path=Config.SUBMISSION_PATH
):
    """
    Formats and saves the submission file according to competition requirements.

    Applies the required probability clipping to avoid log-loss extremes:
    p = max(min(p, 1 - 10^-15), 10^-15)

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (numpy.ndarray): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of probabilities.
        output_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Clip probabilities as per metric description to avoid log(0)
    # Range: [1e-15, 1 - 1e-15]
    clip_epsilon = Config.PROB_CLIP
    clipped_probs = np.clip(probabilities, clip_epsilon, 1.0 - clip_epsilon)

    # Create DataFrame
    # Structure: id, Class1, Class2, ...
    df_submission = pd.DataFrame(clipped_probs, columns=class_names)
    df_submission.insert(0, "id", ids)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {df_submission.shape}")
