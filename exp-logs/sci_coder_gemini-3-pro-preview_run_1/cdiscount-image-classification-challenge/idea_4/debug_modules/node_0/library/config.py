import os
import random
import numpy as np
import torch
import pandas as pd

# ==========================================
# 1. Reproducibility
# ==========================================
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Apply seed immediately upon import
seed_everything()

# ==========================================
# 2. File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# BSON Files
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")

# Metadata Files
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
HIERARCHY_CACHE_PATH = os.path.join(WORKING_DIR, "hierarchy_mappings.parquet")

# ==========================================
# 3. BSON Constants
# ==========================================
BSON_TYPE_DOUBLE = 1
BSON_TYPE_STRING = 2
BSON_TYPE_OBJECT = 3
BSON_TYPE_ARRAY = 4
BSON_TYPE_BINARY = 5
BSON_TYPE_OBJECTID = 7
BSON_TYPE_BOOL = 8
BSON_TYPE_DATE = 9
BSON_TYPE_NULL = 10
BSON_TYPE_INT32 = 16
BSON_TYPE_INT64 = 18

# ==========================================
# 4. Hyperparameters
# ==========================================
BATCH_SIZE = 512
NUM_WORKERS = 12
EPOCHS = 3
LEARNING_RATE = 0.01  # Base LR, intended to be scaled or managed by OneCycleLR
WEIGHT_DECAY = 1e-4
IMG_SIZE = 180

# Hierarchical Loss Configuration
LOSS_WEIGHTS = {"fine": 1.0, "mid": 0.3, "coarse": 0.1}
LABEL_SMOOTHING = 0.1


# ==========================================
# 5. Hierarchy & Data Processing
# ==========================================
def get_hierarchy_mappings(load_cached_data=True):
    """
    Loads category names and generates mappings for hierarchical classification.
    Caches the result to parquet to ensure consistency and speed.
    """
    if load_cached_data and os.path.exists(HIERARCHY_CACHE_PATH):
        print(f"Loading hierarchy mappings from cache: {HIERARCHY_CACHE_PATH}")
        df_mapping = pd.read_parquet(HIERARCHY_CACHE_PATH)
    else:
        print("Computing hierarchy mappings from scratch...")
        if not os.path.exists(CATEGORY_NAMES):
            raise FileNotFoundError(
                f"Category names file not found at {CATEGORY_NAMES}"
            )

        df_cats = pd.read_csv(CATEGORY_NAMES)

        # Sort by category_id to ensure deterministic ordering
        df_cats = df_cats.sort_values("category_id").reset_index(drop=True)

        # Assign continuous index for the fine-grained class (0 to N-1)
        df_cats["class_idx"] = df_cats.index.astype(int)

        # Factorize Level 1 and Level 2 names to get integer labels
        # sort=True ensures alphabetical ordering for determinism
        df_cats["l1_idx"], _ = pd.factorize(df_cats["category_level1"], sort=True)
        df_cats["l2_idx"], _ = pd.factorize(df_cats["category_level2"], sort=True)

        # Save to cache
        df_cats.to_parquet(HIERARCHY_CACHE_PATH)
        df_mapping = df_cats

    # Calculate class counts
    num_classes_l3 = len(df_mapping)
    num_classes_l2 = df_mapping["l2_idx"].max() + 1
    num_classes_l1 = df_mapping["l1_idx"].max() + 1

    # Create lookup dictionaries
    # category_id (int) -> class_idx (int)
    cat_to_idx = dict(zip(df_mapping["category_id"], df_mapping["class_idx"]))

    # class_idx (int) -> category_id (int)
    idx_to_cat = dict(zip(df_mapping["class_idx"], df_mapping["category_id"]))

    # class_idx (int) -> l1_idx (int), l2_idx (int)
    # Using series to_dict for efficiency
    idx_to_l1 = df_mapping.set_index("class_idx")["l1_idx"].to_dict()
    idx_to_l2 = df_mapping.set_index("class_idx")["l2_idx"].to_dict()

    return {
        "df": df_mapping,
        "cat_to_idx": cat_to_idx,
        "idx_to_cat": idx_to_cat,
        "idx_to_l1": idx_to_l1,
        "idx_to_l2": idx_to_l2,
        "num_classes_l1": int(num_classes_l1),
        "num_classes_l2": int(num_classes_l2),
        "num_classes_l3": int(num_classes_l3),
    }
