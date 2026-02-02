import os
import gc
import numpy as np
import pandas as pd
from typing import List, Optional

from library.config import WORKING_DIR, SEED
from library.utils import seed_everything, generate_cache_key
from library.model_factory import UnifiedEnsemble


class MiningCurriculum:
    """
    Orchestrates the Hard Negative Mining curriculum for the IKS-ME strategy.
    Manages Scout training, Hard Negative identification, and Expert dataset construction.
    """

    def __init__(self, feature_cols: List[str], target_col: str = "contact"):
        """
        Initialize the mining curriculum.

        Args:
            feature_cols: List of feature names to be used for training.
            target_col: Name of the target variable.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        seed_everything(SEED)

    def prepare_scout_dataset(
        self, df: pd.DataFrame, neg_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        Prepares a balanced dataset for the Scout model.
        Selects all positives and a random sample of negatives based on neg_ratio.

        Args:
            df: The source dataframe (gated training data).
            neg_ratio: Ratio of negatives to positives (default 1.0 for 1:1 balance).

        Returns:
            Balanced dataframe for scout training.
        """
        pos_mask = df[self.target_col] == 1
        neg_mask = df[self.target_col] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg = int(n_pos * neg_ratio)

        # Clamp n_neg to available negatives
        n_neg = min(n_neg, len(df_neg))

        print(f"Preparing Scout Dataset: {n_pos} Positives, {n_neg} Negatives.")

        # Sample negatives
        df_neg_sampled = df_neg.sample(n=n_neg, random_state=SEED)

        # Combine and shuffle
        df_scout = pd.concat([df_pos, df_neg_sampled])
        df_scout = df_scout.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        return df_scout

    def run_scout_mining(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        hard_neg_threshold: float = 0.05,
        load_cached_mining: bool = True,
    ) -> np.ndarray:
        """
        Trains the Scout model and mines Hard Negatives from the full training set.

        Args:
            df_train: Full gated training dataframe.
            df_val: Validation dataframe.
            hard_neg_threshold: Probability threshold to classify a negative as 'Hard'.
            load_cached_mining: Whether to load indices from cache if available.

        Returns:
            Numpy array of integer indices (iloc) of Hard Negatives in df_train.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Generate cache key based on data dimensions, threshold, and features
        cache_params = {
            "op": "scout_mining",
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "threshold": hard_neg_threshold,
            "features": self.feature_cols,
        }
        cache_key = generate_cache_key(cache_params)
        cache_path = os.path.join(WORKING_DIR, f"hard_negative_indices_{cache_key}.npy")

        # 1. Try Load Cache
        if load_cached_mining and os.path.exists(cache_path):
            print(f"Loading cached Hard Negative indices from {cache_path}...")
            return np.load(cache_path)

        print("Cache miss or force recompute. Starting Scout Mining...")

        # 2. Prepare Scout Data
        df_scout = self.prepare_scout_dataset(df_train)

        # 3. Train Scout Model
        print("Training Scout Ensemble...")
        scout_model = UnifiedEnsemble()
        scout_model.fit(df_scout, df_val, self.feature_cols, self.target_col)

        # Save Scout models with a specific suffix so they don't get confused with Expert models later
        scout_model.save_models(suffix="_scout")

        # 4. Predict on Full Training Set
        print(f"Running Scout Inference on {len(df_train)} training rows...")
        preds = scout_model.predict(df_train, self.feature_cols)

        # 5. Identify Hard Negatives
        # Hard Negative: Label=0 AND Prob > Threshold
        actuals = df_train[self.target_col].values
        hard_neg_mask = (actuals == 0) & (preds > hard_neg_threshold)
        hard_neg_indices = np.where(hard_neg_mask)[0]

        print(f"Mining Complete. Found {len(hard_neg_indices)} Hard Negatives.")

        # 6. Save Cache
        np.save(cache_path, hard_neg_indices)
        print(f"Saved Hard Negative indices to {cache_path}")

        # Cleanup to free memory
        del scout_model, preds, df_scout
        gc.collect()

        return hard_neg_indices

    def prepare_expert_dataset(
        self,
        df_train: pd.DataFrame,
        hard_neg_indices: np.ndarray,
        buffer_ratio: float = 0.1,
    ) -> pd.DataFrame:
        """
        Constructs the Expert Dataset using Positives, Hard Negatives, and a Random Buffer.

        Args:
            df_train: Full gated training dataframe.
            hard_neg_indices: Array of indices for hard negatives.
            buffer_ratio: Ratio of buffer size relative to hard negatives count.

        Returns:
            The Expert training dataframe.
        """
        print("Constructing Expert Dataset...")

        # 1. Identify Indices
        # Positives
        pos_indices = np.where(df_train[self.target_col] == 1)[0]

        # Hard Negatives (ensure they are within bounds)
        hard_neg_indices = hard_neg_indices[hard_neg_indices < len(df_train)]

        # Random Buffer (Easy Negatives)
        # We want negatives that are NOT in hard_neg_indices
        # To do this efficiently, we use set difference on indices
        neg_indices = np.where(df_train[self.target_col] == 0)[0]

        # Set difference: All Negatives - Hard Negatives
        easy_neg_indices = np.setdiff1d(
            neg_indices, hard_neg_indices, assume_unique=True
        )

        # Calculate buffer size
        n_buffer = int(len(hard_neg_indices) * buffer_ratio)
        # Enforce a minimum buffer size (e.g., 2000) to maintain some distribution breadth
        # unless we don't have enough easy negatives
        n_buffer = max(n_buffer, 2000)
        n_buffer = min(n_buffer, len(easy_neg_indices))

        if n_buffer > 0:
            buffer_indices = np.random.choice(
                easy_neg_indices, size=n_buffer, replace=False
            )
        else:
            buffer_indices = np.array([], dtype=int)

        # 2. Combine Indices
        expert_indices = np.concatenate([pos_indices, hard_neg_indices, buffer_indices])
        expert_indices = np.unique(expert_indices)  # Sorts and dedups

        # 3. Create DataFrame
        df_expert = df_train.iloc[expert_indices].copy()

        # 4. Shuffle
        df_expert = df_expert.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        print(f"Expert Dataset Stats:")
        print(f"  Total Rows: {len(df_expert)}")
        print(f"  Positives: {len(pos_indices)}")
        print(f"  Hard Negatives: {len(hard_neg_indices)}")
        print(f"  Buffer Negatives: {len(buffer_indices)}")

        return df_expert
