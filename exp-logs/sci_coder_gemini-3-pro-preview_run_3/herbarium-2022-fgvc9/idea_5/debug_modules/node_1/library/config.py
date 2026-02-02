import os
import json
import pandas as pd
import torch


class Config:
    # =========================================================================
    # 1. Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Output directory for checkpoints, logs, and cached data
    OUTPUT_DIR = "./working/idea_5"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # Submission
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    # Caching
    HIERARCHY_CACHE_PATH = os.path.join(OUTPUT_DIR, "hierarchy_mappings.parquet")

    # =========================================================================
    # 2. Compute & System
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use if DEBUG is True

    # =========================================================================
    # 3. Model Hyperparameters
    # =========================================================================
    # Backbone
    MODEL_NAME = "tf_efficientnetv2_s.in1k"
    PRETRAINED = True

    # Multi-Task Heads (Counts derived from analysis)
    NUM_CLASSES_SPECIES = 15501
    NUM_CLASSES_GENUS = 2564
    NUM_CLASSES_FAMILY = 272

    # Loss Weights
    LOSS_WEIGHT_SPECIES = 1.0
    LOSS_WEIGHT_GENUS = 0.1
    LOSS_WEIGHT_FAMILY = 0.1

    # Regularization
    LABEL_SMOOTHING = 0.1
    DROPOUT_RATE = 0.2
    DROP_PATH_RATE = 0.1

    # =========================================================================
    # 4. Training Strategy (Progressive Resizing)
    # =========================================================================
    # Stage 1: Feature Learning (Low Res, High Throughput)
    STAGE1_IMAGE_SIZE = 224
    STAGE1_BATCH_SIZE = 128
    STAGE1_EPOCHS = 12
    STAGE1_LR = 1e-3

    # Stage 2: Fine-Grained Refinement (High Res)
    STAGE2_IMAGE_SIZE = 320
    STAGE2_BATCH_SIZE = 64
    STAGE2_EPOCHS = 8
    STAGE2_LR = 1e-4

    # Optimization
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP_VAL = 1.0

    # =========================================================================
    # 5. Data Processing Methods
    # =========================================================================
    @staticmethod
    def get_hierarchy_mappings(load_cached_data=True):
        """
        Generates or loads the mapping from category_id to genus_id and family_id.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: DataFrame containing ['category_id', 'genus_id', 'family_id']
                          and mappings for genus/family names.
        """
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        cache_path = Config.HIERARCHY_CACHE_PATH

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading hierarchy mappings from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Generate from scratch
        print("Generating hierarchy mappings from raw metadata...")

        if not os.path.exists(Config.TRAIN_METADATA_JSON):
            raise FileNotFoundError(
                f"Raw metadata not found at {Config.TRAIN_METADATA_JSON}"
            )

        with open(Config.TRAIN_METADATA_JSON, "r") as f:
            meta = json.load(f)

        # Extract categories list
        if "categories" not in meta:
            raise KeyError("Key 'categories' not found in train_metadata.json")

        categories = meta["categories"]
        df_cats = pd.DataFrame(categories)

        # Normalize identifier column to category_id
        if "id" in df_cats.columns:
            df_cats = df_cats.rename(columns={"id": "category_id"})

        # Ensure required columns exist
        required_cols = ["category_id", "genus", "family"]
        for col in required_cols:
            if col not in df_cats.columns:
                # Cite debug_lesson_4: Check if DataFrame is empty to avoid confusing KeyErrors on empty data
                if df_cats.empty:
                    raise ValueError(
                        "Categories DataFrame is empty. Check train_metadata.json content."
                    )
                raise KeyError(
                    f"Column '{col}' missing from categories metadata. Available columns: {list(df_cats.columns)}"
                )

        # Create integer mappings for Genus and Family
        # Sort unique values to ensure deterministic mapping
        unique_genera = sorted(df_cats["genus"].unique())
        unique_families = sorted(df_cats["family"].unique())

        genus_map = {name: i for i, name in enumerate(unique_genera)}
        family_map = {name: i for i, name in enumerate(unique_families)}

        # Map to IDs
        df_cats["genus_id"] = df_cats["genus"].map(genus_map)
        df_cats["family_id"] = df_cats["family"].map(family_map)

        # Select relevant columns
        result_df = df_cats[["category_id", "genus_id", "family_id", "genus", "family"]]

        # Validate counts match Config
        assert (
            len(unique_genera) == Config.NUM_CLASSES_GENUS
        ), f"Genus count mismatch: Found {len(unique_genera)}, Config has {Config.NUM_CLASSES_GENUS}"
        assert (
            len(unique_families) == Config.NUM_CLASSES_FAMILY
        ), f"Family count mismatch: Found {len(unique_families)}, Config has {Config.NUM_CLASSES_FAMILY}"

        # 3. Save to cache
        print(f"Saving hierarchy mappings to {cache_path}")
        result_df.to_parquet(cache_path, index=False)

        return result_df
