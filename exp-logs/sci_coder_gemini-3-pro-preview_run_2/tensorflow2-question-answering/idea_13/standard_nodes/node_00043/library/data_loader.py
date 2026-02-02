import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.vocab_manager import VocabManager
from library.window_processor import WindowProcessor


class WindowNQDataset(Dataset):
    """
    PyTorch Dataset for Window-Based NQ Model.
    Handles negative sampling for training and deterministic loading for evaluation.
    """

    def __init__(
        self, features_df: pd.DataFrame, split: str = "train", config: Config = Config
    ):
        self.split = split
        self.config = config

        # Keep metadata columns for evaluation reconstruction
        self.meta_columns = [
            "example_id",
            "candidate_index",
            "window_index",
            "global_start",
            "global_end",
        ]
        self.metadata = features_df[self.meta_columns].copy().reset_index(drop=True)

        # Convert features to efficient structures
        # Note: input_ids and question_ids are lists in the dataframe.
        # We assume they are already padded by WindowProcessor.
        self.input_ids = list(features_df["input_ids"])
        self.question_ids = list(features_df["question_ids"])

        # Labels
        if split != "test":
            self.label_window = features_df["label_window"].values.astype(np.float32)
            self.label_start = features_df["label_start"].values.astype(np.int64)
            self.label_end = features_df["label_end"].values.astype(np.int64)
            self.label_yes_no = features_df["label_yes_no"].values.astype(np.int64)
        else:
            # Dummy labels for test
            n = len(features_df)
            self.label_window = np.zeros(n, dtype=np.float32)
            self.label_start = np.zeros(n, dtype=np.int64)
            self.label_end = np.zeros(n, dtype=np.int64)
            self.label_yes_no = np.zeros(n, dtype=np.int64)

        # Training: Setup Negative Sampling
        if self.split == "train":
            self.pos_indices = np.where(self.label_window == 1)[0]
            self.neg_indices = np.where(self.label_window == 0)[0]

            self.num_pos = len(self.pos_indices)
            self.num_neg_total = len(self.neg_indices)

            # Determine epoch size based on ratio
            self.num_neg_sample = int(
                self.num_pos * self.config.NEGATIVE_SAMPLING_RATIO
            )
            self.length = self.num_pos + self.num_neg_sample

            # Check if we have enough negatives (unlikely to fail in NQ, but good for safety)
            if self.num_neg_total == 0:
                print("Warning: No negative samples found in training set.")
                self.length = self.num_pos
        else:
            self.length = len(features_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = idx

        if self.split == "train":
            # Dynamic Negative Sampling
            # First N indices are always positive examples (shuffled by DataLoader)
            # Remaining indices are randomly sampled negatives
            if idx < self.num_pos:
                real_idx = self.pos_indices[idx]
            else:
                # Randomly sample a negative index
                # We use numpy's random choice which is seeded globally if np.random.seed is set
                real_idx = np.random.choice(self.neg_indices)

        # Prepare Tensors
        item = {
            "input_ids": torch.tensor(self.input_ids[real_idx], dtype=torch.long),
            "question_ids": torch.tensor(self.question_ids[real_idx], dtype=torch.long),
            "label_window": torch.tensor(
                self.label_window[real_idx], dtype=torch.float
            ),
            "label_start": torch.tensor(self.label_start[real_idx], dtype=torch.long),
            "label_end": torch.tensor(self.label_end[real_idx], dtype=torch.long),
            "label_yes_no": torch.tensor(self.label_yes_no[real_idx], dtype=torch.long),
        }

        # Add metadata (not tensors, handled by custom collate or just list of strings in default collate)
        # Note: Default collate in PyTorch > 1.2 handles strings by batching them into lists
        meta_row = self.metadata.iloc[real_idx]
        item["example_id"] = str(meta_row["example_id"])
        item["candidate_index"] = int(meta_row["candidate_index"])
        item["window_index"] = int(meta_row["window_index"])
        item["global_start"] = int(meta_row["global_start"])
        item["global_end"] = int(meta_row["global_end"])

        return item


def normalize_id(x):
    """
    Normalizes example_id to ensure consistent string format.
    Handles cases where IDs might be represented as floats (e.g., '123.0') in metadata.
    Cite debug_lesson_12: Enforce String Typing for Identifiers.
    """
    s = str(x).strip()
    if s.endswith(".0"):
        return s[:-2]
    return s


def get_data_loaders(
    config: Config, vocab_manager: VocabManager, load_cached_data: bool = True
):
    """
    Generates DataLoaders for train, validation, and test sets.
    Uses WindowProcessor to generate features and Metadata files to split them.
    """
    print(f"Initializing WindowProcessor...")
    processor = WindowProcessor(config, vocab_manager)

    # --- 1. Load/Process Features ---
    # Train features (contains both train and validation splits based on source file)
    print("Loading training source features...")
    full_train_features = processor.process_dataset(
        load_cached_data=load_cached_data, is_train=True
    )

    # Test features
    print("Loading test source features...")
    test_features = processor.process_dataset(
        load_cached_data=load_cached_data, is_train=False
    )

    # --- 2. Load Metadata for Splitting ---
    print("Loading metadata for splitting...")
    if not os.path.exists(config.TRAIN_META_PATH) or not os.path.exists(
        config.VAL_META_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation script has run."
        )

    # Fix: Enforce string dtype for IDs to prevent mismatch with JSON features
    # Cite debug_lesson_12
    train_meta = pd.read_csv(config.TRAIN_META_PATH, dtype={"example_id": str})
    val_meta = pd.read_csv(config.VAL_META_PATH, dtype={"example_id": str})

    # Get sets of example_ids
    # Normalize IDs to handle potential ".0" suffixes from previous float inferences
    train_ids = set(train_meta["example_id"].apply(normalize_id))
    val_ids = set(val_meta["example_id"].apply(normalize_id))

    # --- 3. Split Training Data ---
    print("Splitting features into Train and Validation sets...")
    # Filter using boolean indexing
    # Ensure example_id column in features is string and normalized
    full_train_features["example_id"] = full_train_features["example_id"].apply(
        normalize_id
    )

    train_df = full_train_features[
        full_train_features["example_id"].isin(train_ids)
    ].copy()
    val_df = full_train_features[full_train_features["example_id"].isin(val_ids)].copy()

    print(f"Total Train Windows: {len(train_df)}")
    print(f"Total Validation Windows: {len(val_df)}")
    print(f"Total Test Windows: {len(test_features)}")

    # --- 4. Create Datasets ---
    train_dataset = WindowNQDataset(train_df, split="train", config=config)
    val_dataset = WindowNQDataset(val_df, split="val", config=config)
    test_dataset = WindowNQDataset(test_features, split="test", config=config)

    # --- 5. Create DataLoaders ---
    # Num workers set to 0 for safety in constrained environments, can be increased if supported
    num_workers = 2 if os.cpu_count() > 2 else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
