import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from library.config import Config


def seed_everything(seed: int = Config.seed):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cpc_texts(path: str = Config.cpc_codes_path) -> dict:
    """
    Returns a dictionary mapping CPC codes to their full textual descriptions.

    Args:
        path (str): Path to the CPC titles CSV file. Defaults to Config.cpc_codes_path.

    Returns:
        dict: A dictionary where keys are CPC codes (e.g., 'A47') and values are
              their descriptions (e.g., 'Furniture...'). Returns an empty dict if
              the file is not found.
    """
    cpc_texts = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Expecting 'code' and 'title' columns in the external dataset
            if "code" in df.columns and "title" in df.columns:
                cpc_texts = dict(zip(df["code"], df["title"]))
        except Exception:
            # If reading fails, return empty dict to allow pipeline to proceed without enrichment
            pass
    return cpc_texts


def compute_pearson_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (np.ndarray): Array of ground truth scores.
        y_pred (np.ndarray): Array of predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    score, _ = pearsonr(y_true, y_pred)
    return score
