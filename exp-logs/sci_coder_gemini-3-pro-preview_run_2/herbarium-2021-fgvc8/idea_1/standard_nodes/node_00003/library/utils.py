import os
import json
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score


def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.
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


def calculate_f1_score(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


def get_taxonomy_mappings(
    metadata_path="./input/train/metadata.json", load_cached_data=True
):
    """
    Parses metadata to create mappings for Species, Family, and Order.
    Handles caching to parquet to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the training metadata JSON.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (
            species_to_family (dict): category_id -> family_id (int),
            species_to_order (dict): category_id -> order_id (int),
            species_to_idx (dict): category_id -> class_idx (0..N-1),
            idx_to_species (dict): class_idx -> category_id,
            num_families (int),
            num_orders (int),
            num_species (int)
        )
    """
    cache_dir = "./working/idea_1"
    cache_file = os.path.join(cache_dir, "taxonomy_mappings.parquet")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)

            # Reconstruct dictionaries from dataframe
            species_to_family = dict(zip(df["category_id"], df["family_id"]))
            species_to_order = dict(zip(df["category_id"], df["order_id"]))
            species_to_idx = dict(zip(df["category_id"], df["class_idx"]))
            idx_to_species = dict(zip(df["class_idx"], df["category_id"]))

            num_families = df["family_id"].max() + 1
            num_orders = df["order_id"].max() + 1
            num_species = df["class_idx"].max() + 1

            return (
                species_to_family,
                species_to_order,
                species_to_idx,
                idx_to_species,
                int(num_families),
                int(num_orders),
                int(num_species),
            )
        except Exception:
            # Fallback to processing if cache load fails
            pass

    # Process from scratch
    with open(metadata_path, "r") as f:
        data = json.load(f)

    categories = data["categories"]

    # Sort categories by ID to ensure deterministic ordering
    categories.sort(key=lambda x: x["id"])

    # 1. Handle Species Mapping (category_id -> 0..N-1)
    species_ids = [c["id"] for c in categories]
    species_to_idx = {sid: i for i, sid in enumerate(species_ids)}
    idx_to_species = {i: sid for sid, i in species_to_idx.items()}
    num_species = len(species_ids)

    # 2. Handle Family Mapping (string -> int)
    unique_families = sorted(list(set(c["family"] for c in categories)))
    family_map = {fam: i for i, fam in enumerate(unique_families)}
    num_families = len(unique_families)

    # 3. Handle Order Mapping (string -> int)
    unique_orders = sorted(list(set(c["order"] for c in categories)))
    order_map = {ordr: i for i, ordr in enumerate(unique_orders)}
    num_orders = len(unique_orders)

    # 4. Create Species -> Family/Order ID mappings
    species_to_family = {}
    species_to_order = {}

    data_rows = []

    for c in categories:
        cat_id = c["id"]
        fam_id = family_map[c["family"]]
        ord_id = order_map[c["order"]]
        cls_idx = species_to_idx[cat_id]

        species_to_family[cat_id] = fam_id
        species_to_order[cat_id] = ord_id

        data_rows.append(
            {
                "category_id": cat_id,
                "family_id": fam_id,
                "order_id": ord_id,
                "class_idx": cls_idx,
            }
        )

    # Save to cache
    df = pd.DataFrame(data_rows)
    df.to_parquet(cache_file, index=False)

    return (
        species_to_family,
        species_to_order,
        species_to_idx,
        idx_to_species,
        num_families,
        num_orders,
        num_species,
    )
