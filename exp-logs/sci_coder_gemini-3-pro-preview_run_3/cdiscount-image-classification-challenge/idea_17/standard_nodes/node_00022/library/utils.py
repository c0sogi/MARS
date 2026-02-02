import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CuDNN backend.
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


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Designed to print full precision without rounding.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        # Print full precision as requested
        return f"{self.name} {self.val} ({self.avg})"


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_hierarchy_mappings(load_cached_data=True):
    """
    Generates or loads mappings for hierarchical categories.

    Logic:
    1. Checks if cached parquet file exists in working directory.
    2. If yes and load_cached_data is True, loads it.
    3. If no, reads category_names.csv, builds mappings for L1, L2, L3 levels.
    4. Saves the mapping DataFrame to parquet for future use.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        mapping_dict (dict): Maps category_id -> {'l1': idx, 'l2': idx, 'l3': idx}
        idx_to_category_id (dict): Maps l3_idx -> category_id (for submission generation)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = Config.HIERARCHY_MAPPING_PATH

    df = None

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # If load fails, proceed to recompute
            df = None

    # 2. Compute if not loaded
    if df is None:
        cats = pd.read_csv(Config.CATEGORY_NAMES)

        # Level 1 Mapping (Coarse)
        l1_names = sorted(cats["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_names)}

        # Level 2 Mapping (Sub-category)
        l2_names = sorted(cats["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_names)}

        # Level 3 Mapping (Fine-grained / category_id)
        # Sort by category_id to ensure deterministic index assignment
        l3_ids = sorted(cats["category_id"].unique())
        l3_map = {cid: i for i, cid in enumerate(l3_ids)}

        # Build DataFrame for caching
        data = []
        for _, row in cats.iterrows():
            cid = row["category_id"]
            l1 = row["category_level1"]
            l2 = row["category_level2"]

            data.append(
                {
                    "category_id": cid,
                    "l1_idx": l1_map[l1],
                    "l2_idx": l2_map[l2],
                    "l3_idx": l3_map[cid],
                }
            )

        df = pd.DataFrame(data)

        # Save to cache using parquet
        df.to_parquet(cache_path, index=False)

    # 3. Convert to efficient dictionaries
    mapping_dict = {}
    idx_to_category_id = {}

    # Using itertuples for speed
    for row in df.itertuples(index=False):
        cid = int(row.category_id)
        l1 = int(row.l1_idx)
        l2 = int(row.l2_idx)
        l3 = int(row.l3_idx)

        # Map raw ID to hierarchical indices
        mapping_dict[cid] = {"l1": l1, "l2": l2, "l3": l3}

        # Map L3 index back to raw ID (for submission)
        idx_to_category_id[l3] = cid

    return mapping_dict, idx_to_category_id
