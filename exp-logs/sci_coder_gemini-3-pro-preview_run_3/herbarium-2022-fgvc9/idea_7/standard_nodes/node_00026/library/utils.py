import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="train"):
    """
    Creates and returns a logger that prints to stdout.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def calculate_macro_f1(y_true, y_pred):
    """
    Computes the Macro F1 score.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels.
        y_pred (np.array or torch.Tensor): Predicted labels.

    Returns:
        float: Macro F1 score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    return f1_score(y_true, y_pred, average="macro")


def load_hierarchy_mappings(json_path, cache_path, load_cached_data=True):
    """
    Loads taxonomic hierarchy (Category -> Genus -> Family) from metadata JSON.
    Implements caching using Parquet to ensure deterministic processing and speed.

    Args:
        json_path (str): Path to the train_metadata.json file.
        cache_path (str): Path where the processed dataframe should be cached (parquet).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with columns ['category_id', 'genus_id', 'family_id', 'genus', 'family'].
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

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
        raise FileNotFoundError(f"Metadata JSON not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract categories
    # Structure: {"categories": [{"id": 0, "family": "...", "genus": "...", "species": "..."}, ...]}
    if "categories" not in data:
        raise ValueError("JSON does not contain 'categories' key.")

    categories = data["categories"]
    df = pd.DataFrame(categories)

    # Rename 'id' to 'category_id' for consistency if needed
    if "id" in df.columns:
        df.rename(columns={"id": "category_id"}, inplace=True)

    # Ensure category_id is int
    df["category_id"] = df["category_id"].astype(int)

    # Encode Family and Genus
    # We sort unique values to ensure deterministic encoding (0 to N-1 based on alphabetical order)
    unique_families = sorted(df["family"].unique())
    unique_genera = sorted(df["genus"].unique())

    family_map = {name: i for i, name in enumerate(unique_families)}
    genus_map = {name: i for i, name in enumerate(unique_genera)}

    df["family_id"] = df["family"].map(family_map)
    df["genus_id"] = df["genus"].map(genus_map)

    # Encode Species (Category ID) to contiguous range 0..N-1
    # Cite debug_lesson_1: Map Sparse or Non-Zero-Based Labels to Contiguous Indices
    unique_categories = sorted(df["category_id"].unique())
    species_map = {cat_id: i for i, cat_id in enumerate(unique_categories)}
    df["species_label"] = df["category_id"].map(species_map)

    # Select relevant columns
    cols = ["category_id", "species_label", "family_id", "genus_id", "family", "genus"]
    df = df[cols]

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        # Proceed without crashing if cache save fails
        pass

    return df
