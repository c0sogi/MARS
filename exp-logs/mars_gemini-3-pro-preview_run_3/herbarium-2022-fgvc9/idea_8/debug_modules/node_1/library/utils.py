import os
import json
import random
import logging
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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
    Creates and configures a logger that outputs to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def calculate_macro_f1(preds, targets):
    """
    Calculates the Macro F1 score.

    Args:
        preds (np.array or torch.Tensor): Predicted labels.
        targets (np.array or torch.Tensor): Ground truth labels.

    Returns:
        float: Macro F1 score.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return f1_score(targets, preds, average="macro")


def load_hierarchy_mappings(
    json_path="./input/train_metadata.json",
    cache_dir="./working/idea_8",
    load_cached_data=True,
):
    """
    Loads hierarchy mappings (Species -> Genus, Species -> Family).
    Implements caching using Parquet.

    Args:
        json_path (str): Path to the training metadata JSON.
        cache_dir (str): Directory to store/load the cached parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing:
            - 'species_to_genus': dict mapping category_id to genus_id
            - 'species_to_family': dict mapping category_id to family_id
            - 'num_genera': int count of unique genera
            - 'num_families': int count of unique families
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "hierarchy_mappings.parquet")

    df = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # print(f"Loaded hierarchy mappings from cache: {cache_file}")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            df = None

    # 2. Compute if not loaded
    if df is None:
        # print(f"Processing hierarchy from {json_path}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract categories. Structure is expected to be a list of dicts under 'categories' key
        # or the root object might be the list itself depending on specific JSON structure.
        # Based on standard COCO-like datasets and description:
        if isinstance(data, dict) and "categories" in data:
            categories = data["categories"]
        elif isinstance(data, list):
            # Fallback if the json is just a list of categories (unlikely but possible)
            categories = data
        else:
            raise ValueError("Unexpected JSON structure in train_metadata.json")

        df = pd.DataFrame(categories)

        # Ensure required columns exist
        required_cols = {"id", "genus", "family"}
        if not required_cols.issubset(df.columns):
            # If 'id' is missing but we have 'category_id', rename it.
            # If names are different, we might need adjustment, but 'id', 'genus', 'family' is standard.
            if "category_id" in df.columns and "id" not in df.columns:
                df = df.rename(columns={"category_id": "id"})
            else:
                raise ValueError(
                    f"Metadata missing required columns. Found: {df.columns}"
                )

        # Encode Genus and Family to integers
        # We sort to ensure deterministic encoding
        df["genus"] = df["genus"].astype(str)
        df["family"] = df["family"].astype(str)

        # Use pandas categorical codes
        # Sort by name first to ensure ID 0 is always the first alphabetically
        unique_genera = sorted(df["genus"].unique())
        unique_families = sorted(df["family"].unique())

        genus_map = {name: i for i, name in enumerate(unique_genera)}
        family_map = {name: i for i, name in enumerate(unique_families)}

        df["genus_id"] = df["genus"].map(genus_map)
        df["family_id"] = df["family"].map(family_map)

        # Keep only necessary columns for the mapping
        df = df[["id", "genus_id", "family_id"]].rename(columns={"id": "category_id"})

        # Save to cache
        df.to_parquet(cache_file)
        # print(f"Saved hierarchy mappings to cache: {cache_file}")

    # 3. Convert to dictionaries for O(1) lookup
    species_to_genus = dict(zip(df["category_id"], df["genus_id"]))
    species_to_family = dict(zip(df["category_id"], df["family_id"]))

    num_genera = df["genus_id"].max() + 1
    num_families = df["family_id"].max() + 1

    return {
        "species_to_genus": species_to_genus,
        "species_to_family": species_to_family,
        "num_genera": int(num_genera),
        "num_families": int(num_families),
    }
