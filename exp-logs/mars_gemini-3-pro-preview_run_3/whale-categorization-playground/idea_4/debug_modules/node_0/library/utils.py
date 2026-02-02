import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = None):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int, optional): The seed to use. If None, uses Config.seed.
    """
    if seed is None:
        seed = Config.seed

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_map5(ground_truth, predictions):
    """
    Computes the Mean Average Precision @ 5 (MAP@5).

    Args:
        ground_truth (list or np.array): List of correct labels (strings).
        predictions (list or np.array): List of lists of predicted labels (strings).
                                        Each inner list should have up to 5 predictions.

    Returns:
        float: The MAP@5 score.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("Length of ground_truth and predictions must match.")

    n = len(ground_truth)
    if n == 0:
        return 0.0

    score_sum = 0.0

    for truth, preds in zip(ground_truth, predictions):
        # Ensure we only consider the top 5 predictions
        top_preds = list(preds)[:5]

        if truth in top_preds:
            # Rank is 0-indexed, so the score is 1 / (rank + 1)
            rank = top_preds.index(truth)
            score_sum += 1.0 / (rank + 1)
        else:
            score_sum += 0.0

    return score_sum / n


def load_metadata(split: str, metadata_dir: str = Config.metadata_dir):
    """
    Loads the metadata CSV for a given split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        metadata_dir (str): Directory where metadata CSVs are stored.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    file_path = os.path.join(metadata_dir, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    return pd.read_csv(file_path)


def save_submission(
    image_names, predictions, output_path: str = Config.submission_path
):
    """
    Saves predictions to a CSV file in the competition format.

    Args:
        image_names (list): List of image filenames.
        predictions (list): List of lists of predicted labels.
        output_path (str): Path to save the submission CSV.
    """
    # Format: Space-separated string of top 5 predictions
    formatted_preds = [" ".join(p[:5]) for p in predictions]

    df = pd.DataFrame({"Image": image_names, "Id": formatted_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
