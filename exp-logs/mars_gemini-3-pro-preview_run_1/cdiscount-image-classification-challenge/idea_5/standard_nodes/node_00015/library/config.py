import os
import pandas as pd
import numpy as np
import torch

# ==========================================
# FILE PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Files
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Files
HIERARCHY_CACHE_PATH = os.path.join(WORKING_DIR, "hierarchy_mappings.parquet")

# ==========================================
# HYPERPARAMETERS
# ==========================================
SEED = 42
BATCH_SIZE = 512  # Target 512 to saturate A100 GPU
NUM_EPOCHS = 4  # Short schedule (3-4 epochs) for massive dataset
LEARNING_RATE = 0.01  # Base LR, intended to be used with OneCycleLR
NUM_WORKERS = 12  # Maximize data loading throughput
IMG_SIZE = 180  # Native image size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss Configuration
# Deep Supervision: L_total = L_level3 + 0.3 * L_level2 + 0.1 * L_level1
LOSS_WEIGHTS = {"level3": 1.0, "level2": 0.3, "level1": 0.1}
LABEL_SMOOTHING = 0.1


# ==========================================
# HIERARCHY MAPPING UTILITIES
# ==========================================
def get_hierarchy_mappings(load_cached_data=True):
    """
    Generates or loads mappings between category_ids and their hierarchical levels.

    Returns:
        mappings_df (pd.DataFrame): DataFrame containing 'category_id', 'l1_idx', 'l2_idx', 'l3_idx'.
        stats (dict): Dictionary containing the number of classes for each level.
    """

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(HIERARCHY_CACHE_PATH):
        try:
            mappings_df = pd.read_parquet(HIERARCHY_CACHE_PATH)

            # Recalculate stats from loaded data
            stats = {
                "num_classes_l1": mappings_df["l1_idx"].max() + 1,
                "num_classes_l2": mappings_df["l2_idx"].max() + 1,
                "num_classes_l3": mappings_df["l3_idx"].max() + 1,
            }
            return mappings_df, stats
        except Exception as e:
            print(f"Failed to load cached hierarchy mappings: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(CATEGORY_NAMES):
        raise FileNotFoundError(f"Category names file not found at {CATEGORY_NAMES}")

    df_cats = pd.read_csv(CATEGORY_NAMES)

    # Ensure no NaNs in critical columns (based on dataset analysis, there are none, but good practice)
    df_cats = df_cats.dropna(
        subset=["category_id", "category_level1", "category_level2"]
    )

    # -- Level 1 Mapping (Coarse) --
    # Sort for determinism
    l1_names = sorted(df_cats["category_level1"].unique())
    l1_map = {name: i for i, name in enumerate(l1_names)}

    # -- Level 2 Mapping (Intermediate) --
    l2_names = sorted(df_cats["category_level2"].unique())
    l2_map = {name: i for i, name in enumerate(l2_names)}

    # -- Level 3 Mapping (Fine / Target) --
    # We map the actual category_id (int) to a 0-indexed range for the classifier
    l3_ids = sorted(df_cats["category_id"].unique())
    l3_map = {cat_id: i for i, cat_id in enumerate(l3_ids)}

    # Apply mappings
    df_cats["l1_idx"] = df_cats["category_level1"].map(l1_map).astype(np.int32)
    df_cats["l2_idx"] = df_cats["category_level2"].map(l2_map).astype(np.int32)
    df_cats["l3_idx"] = df_cats["category_id"].map(l3_map).astype(np.int32)

    # Select relevant columns for the mapping dataframe
    mappings_df = df_cats[["category_id", "l1_idx", "l2_idx", "l3_idx"]].copy()

    # 3. Save to cache
    try:
        mappings_df.to_parquet(HIERARCHY_CACHE_PATH, index=False)
    except Exception as e:
        print(f"Warning: Could not save hierarchy cache: {e}")

    # 4. Generate stats
    stats = {
        "num_classes_l1": len(l1_names),
        "num_classes_l2": len(l2_names),
        "num_classes_l3": len(l3_ids),
    }

    return mappings_df, stats
