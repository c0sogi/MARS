import os
import random
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(log_file):
    """
    Sets up a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def process_hierarchy_mappings(json_path, cache_dir, load_cached_data=True):
    """
    Parses the training metadata JSON to extract taxonomic hierarchy mappings.
    Maps category_id to genus_id and family_id for multi-task learning.

    Args:
        json_path (str): Path to the train_metadata.json file.
        cache_dir (str): Directory to store the cached parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing category_id, genus_id, family_id,
                      genus_name, and family_name.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hierarchy_mappings.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from scratch
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract categories list
    # Expected structure: {'categories': [{'id': 0, 'genus': '...', 'family': '...'}, ...]}
    categories = data.get("categories", [])

    extracted_data = []
    for cat in categories:
        # Check for 'id' or 'category_id' to be robust
        cat_id = cat.get("id", cat.get("category_id"))
        if cat_id is None:
            continue

        extracted_data.append(
            {
                "category_id": cat_id,
                "genus_name": cat.get("genus"),
                "family_name": cat.get("family"),
            }
        )

    # Explicitly define columns to prevent KeyError on empty data
    # Cite {debug_lesson_4}
    df = pd.DataFrame(
        extracted_data, columns=["category_id", "genus_name", "family_name"]
    )

    # Encode Genus and Family names to integer IDs
    # Sorting ensures deterministic encoding across runs
    genus_list = sorted(df["genus_name"].dropna().unique())
    family_list = sorted(df["family_name"].dropna().unique())

    genus_map = {name: i for i, name in enumerate(genus_list)}
    family_map = {name: i for i, name in enumerate(family_list)}

    df["genus_id"] = df["genus_name"].map(genus_map).fillna(-1).astype(int)
    df["family_id"] = df["family_name"].map(family_map).fillna(-1).astype(int)

    # Ensure dataframe is sorted by category_id for consistent indexing
    df = df.sort_values("category_id").reset_index(drop=True)

    # 3. Save to cache
    df.to_parquet(cache_path)

    return df
