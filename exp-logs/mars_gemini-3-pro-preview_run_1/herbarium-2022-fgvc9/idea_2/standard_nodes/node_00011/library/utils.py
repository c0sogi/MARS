import os
import json
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_taxonomy_mappings(
    metadata_json_path="./input/train_metadata.json", load_cached_data=True
):
    """
    Parses metadata to create mappings from category_id to family and genus indices.

    Args:
        metadata_json_path (str): Path to the training metadata JSON file.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (cat_to_fam, cat_to_gen, num_families, num_genera)
            - cat_to_fam (dict): category_id (int) -> family_id (int)
            - cat_to_gen (dict): category_id (int) -> genus_id (int)
            - num_families (int): Total number of unique families
            - num_genera (int): Total number of unique genera
    """
    cache_dir = "./working/idea_2/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_mappings.json")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            # JSON keys are strings, convert back to int for category_id
            cat_to_fam = {int(k): v for k, v in cached["cat_to_fam"].items()}
            cat_to_gen = {int(k): v for k, v in cached["cat_to_gen"].items()}
            return cat_to_fam, cat_to_gen, cached["num_families"], cached["num_genera"]
        except Exception:
            pass  # Proceed to compute if cache load fails

    # Compute mappings
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)

    categories = meta.get("categories", [])

    # Extract unique sorted families and genera to ensure deterministic ID assignment
    unique_families = sorted(list(set(c["family"] for c in categories)))
    unique_genera = sorted(list(set(c["genus"] for c in categories)))

    fam_to_idx = {name: i for i, name in enumerate(unique_families)}
    gen_to_idx = {name: i for i, name in enumerate(unique_genera)}

    cat_to_fam = {}
    cat_to_gen = {}

    for c in categories:
        cid = int(c["category_id"])
        cat_to_fam[cid] = fam_to_idx[c["family"]]
        cat_to_gen[cid] = gen_to_idx[c["genus"]]

    # Save to cache
    to_save = {
        "cat_to_fam": cat_to_fam,
        "cat_to_gen": cat_to_gen,
        "num_families": len(unique_families),
        "num_genera": len(unique_genera),
    }

    with open(cache_path, "w") as f:
        json.dump(to_save, f)

    return cat_to_fam, cat_to_gen, len(unique_families), len(unique_genera)


def compute_loss_weights(
    train_csv_path="./metadata/train.csv", num_classes=15501, load_cached_data=True
):
    """
    Computes inverse frequency weights for class balancing.
    Assumes classes are mapped to 0..num_classes-1 based on sorted category_ids.

    Args:
        train_csv_path (str): Path to the training CSV.
        num_classes (int): Number of unique classes.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing weights.
    """
    cache_dir = "./working/idea_2/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            if weights_np.shape[0] == num_classes:
                return torch.tensor(weights_np, dtype=torch.float32)
        except Exception:
            pass  # Proceed to compute if cache load fails

    # Compute weights
    df = pd.read_csv(train_csv_path)

    # Get counts for each category_id
    # We assume the model maps sorted(unique_category_ids) -> 0..N-1
    unique_cats = sorted(df["category_id"].unique())
    counts_map = df["category_id"].value_counts().to_dict()

    total_samples = len(df)
    weights_list = []

    # Iterate in the sorted order of category_ids to align with label indices
    for cid in unique_cats:
        count = counts_map.get(cid, 0)
        if count > 0:
            # Inverse frequency weight: N / (C * n_c)
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights_list.append(w)

    # Handle case where train.csv might miss some classes (though unlikely with proper metadata)
    if len(weights_list) < num_classes:
        weights_list.extend([1.0] * (num_classes - len(weights_list)))

    weights_np = np.array(weights_list, dtype=np.float32)

    # Save to cache
    np.save(cache_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32)
