import pandas as pd
import numpy as np
import os
import sys
from library.config import Config
from library.utils import get_logger, save_cache, load_cache, ensure_dir
from library.transformations import TransformationRegistry

logger = get_logger("label_manager")


class LabelEngineer:
    """
    Handles the creation of fine-grained transformation labels for the dataset.
    Implements Inverse Label Engineering by matching raw/normalized pairs to
    deterministic functions in the TransformationRegistry.
    """

    def __init__(self):
        self.registry = TransformationRegistry()
        self.label_names = None  # List of label names, index corresponds to ID
        self.name_to_id = None  # Dict mapping name -> ID

    def _load_or_create_label_encoder(self):
        """
        Loads the label encoder from disk or creates it from the registry.
        The encoder is an ordered list of transformation names saved as a .npy file.
        """
        if os.path.exists(Config.LABEL_ENCODER_PATH):
            logger.info(f"Loading label encoder from {Config.LABEL_ENCODER_PATH}")
            self.label_names = np.load(
                Config.LABEL_ENCODER_PATH, allow_pickle=True
            ).tolist()
        else:
            logger.info("Creating new label encoder from registry...")
            # Get all registered function keys
            keys = sorted(list(self.registry.funcs.keys()))

            # Ensure TRANS_PLAIN is at index 0 for consistency/default
            if "TRANS_PLAIN" in keys:
                keys.remove("TRANS_PLAIN")
            keys.insert(0, "TRANS_PLAIN")

            self.label_names = keys

            # Save to disk
            ensure_dir(Config.LABEL_ENCODER_PATH)
            np.save(Config.LABEL_ENCODER_PATH, np.array(self.label_names))
            logger.info(f"Saved label encoder with {len(self.label_names)} classes.")

        # Create look-up dictionary
        self.name_to_id = {name: idx for idx, name in enumerate(self.label_names)}

    def process_dataset(
        self, split: str, load_cached_data: bool = True, debug: bool = Config.DEBUG
    ) -> pd.DataFrame:
        """
        Processes the dataset for the given split (train/val/test).
        Derives labels for 'train' and 'val' splits based on 'before' and 'after' columns.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from parquet cache.
            debug (bool): If True, processes only a small subset and does not save to main cache.

        Returns:
            pd.DataFrame: The processed dataframe with 'label_id' column.
        """
        # Determine file paths based on split
        if split == "train":
            meta_path = Config.TRAIN_METADATA
            cache_path = Config.TRAIN_PROCESSED
        elif split == "val":
            meta_path = Config.VAL_METADATA
            cache_path = Config.VAL_PROCESSED
        elif split == "test":
            meta_path = Config.TEST_METADATA
            cache_path = Config.TEST_PROCESSED
        else:
            raise ValueError(f"Unknown split: {split}")

        # 1. Try loading from cache (only if not debugging, or if we want to debug the cache loading)
        # Typically, if debug is True, we might want to re-run logic on a subset, so we skip cache.
        if load_cached_data and not debug:
            df = load_cache(cache_path)
            if df is not None:
                logger.info(f"Loaded {split} processed data from {cache_path}")
                # Ensure encoder is loaded so we have the mapping available
                self._load_or_create_label_encoder()
                return df

        # 2. Process from scratch
        logger.info(f"Processing {split} data from scratch (Debug={debug})...")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load raw metadata
        df = pd.read_csv(meta_path, keep_default_na=False)

        # Apply Debug Slicing
        if debug:
            logger.info("Debug mode: Slicing dataset to first 10,000 rows.")
            df = df.head(10000).copy()

        # Ensure label encoder is ready
        # For training set, we might create it. For others, we load it.
        # Since we want a consistent encoder across all splits, we usually create it once based on registry.
        self._load_or_create_label_encoder()

        # 3. Derive Labels
        if split in ["train", "val"]:
            logger.info("Deriving fine-grained transformation labels...")

            # Extract arrays for faster iteration
            befores = df["before"].astype(str).values
            afters = df["after"].astype(str).values
            classes = df["class"].astype(str).values

            labels = []
            find_transform = self.registry.find_best_transform

            # Iterate through all rows
            # Note: We avoid tqdm as it is not in the allowed package list
            total = len(df)
            log_interval = max(1, total // 10)

            for i, (b, a, c) in enumerate(zip(befores, afters, classes)):
                if i % log_interval == 0 and i > 0:
                    logger.info(f"Processed {i}/{total} rows...")

                lbl = find_transform(b, a, c)
                labels.append(lbl)

            df["label_name"] = labels

            # Map string labels to integer IDs
            # Default to TRANS_PLAIN (ID 0) if something unexpected happens
            default_id = self.name_to_id.get("TRANS_PLAIN", 0)
            df["label_id"] = (
                df["label_name"].map(self.name_to_id).fillna(default_id).astype(int)
            )

            # Log distribution statistics
            logger.info(f"Label distribution for {split} (Top 10):")
            logger.info(f"\n{df['label_name'].value_counts().head(10)}")

        else:
            # Test set: No 'after' column, so we cannot derive labels.
            # Initialize with dummy labels (PLAIN) for compatibility with Dataset classes.
            logger.info("Test set: Initializing dummy labels.")
            df["label_id"] = self.name_to_id.get("TRANS_PLAIN", 0)
            df["label_name"] = "TRANS_PLAIN"

        # 4. Save to cache (only if not debugging)
        if not debug:
            save_cache(df, cache_path)
            logger.info(f"Saved processed data to {cache_path}")
        else:
            logger.info("Debug mode: Skipping cache save.")

        return df
