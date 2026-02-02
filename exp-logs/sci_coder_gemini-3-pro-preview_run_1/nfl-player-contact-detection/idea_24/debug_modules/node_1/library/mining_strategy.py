import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import (
    seed_everything,
    gaussian_smooth_labels,
    save_cache,
    load_cache,
)
from library.model_factory import LGBMExpert, XGBExpert


class MiningStrategy:
    """
    Implements the Tri-Scout Anchored Mining Curriculum.

    Phases:
    1. Train Scouts: Train LGBM, XGB, and CatBoost on a balanced subset of training data.
    2. Mine Hard Negatives: Use Scouts to identify negatives with P(Contact) > Threshold.
    3. Construct Anchored Dataset: Combine Positives, Hard Negatives, and Random Anchors
       with Temporal Label Smoothing for the final Expert training.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.features = Config.FEATURES

    def train_scouts(self, df_train, df_val=None):
        """
        Phase 1: Train Scout models on a balanced subset of the gated survivors.

        Args:
            df_train (pd.DataFrame): The training data (gated).
            df_val (pd.DataFrame, optional): Validation data for early stopping.

        Returns:
            dict: A dictionary containing the trained scout models.
        """
        print("\n--- Phase 1: Training Tri-Scout Ensemble (Balanced) ---")

        # 1. Create Balanced Training Set
        # Separate positives and negatives
        df_pos = df_train[df_train["contact"] == 1]
        df_neg = df_train[df_train["contact"] == 0]

        # Downsample negatives to match positives (1:1 ratio for Scouts)
        n_pos = len(df_pos)
        if len(df_neg) > n_pos:
            df_neg_balanced = df_neg.sample(n=n_pos, random_state=Config.SEED)
        else:
            df_neg_balanced = df_neg

        df_balanced = (
            pd.concat([df_pos, df_neg_balanced])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        print(
            f"Balanced Training Set: {len(df_balanced)} samples ({len(df_pos)} Pos, {len(df_neg_balanced)} Neg)"
        )

        X_train = df_balanced[self.features]
        y_train = df_balanced["contact"]

        X_val_feats = df_val[self.features] if df_val is not None else None
        y_val_labels = df_val["contact"] if df_val is not None else None

        # 2. Train Scouts
        scouts = {}

        # Scout A: LightGBM
        print("Training Scout A (LightGBM)...")
        scout_lgbm = LGBMExpert()
        scout_lgbm.fit(X_train, y_train, X_val_feats, y_val_labels)
        scouts["lgbm"] = scout_lgbm

        # Scout B: XGBoost
        print("Training Scout B (XGBoost)...")
        scout_xgb = XGBExpert()
        scout_xgb.fit(X_train, y_train, X_val_feats, y_val_labels)
        scouts["xgb"] = scout_xgb

        return scouts

    def mine_hard_negatives(self, df_train, scouts, load_cached_data=True):
        """
        Phase 2: Diversity Mining.
        Identifies 'Hard Negatives' in the full training set where any Scout predicts > Threshold.

        Args:
            df_train (pd.DataFrame): The full training data to mine.
            scouts (dict): Dictionary of trained scout models.
            load_cached_data (bool): Whether to load indices from cache.

        Returns:
            np.ndarray: Array of indices (relative to df_train) representing hard negatives.
        """
        cache_path = Config.CACHE_HARD_NEGATIVES

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached hard negative indices from {cache_path}...")
            return load_cache(cache_path)

        print("\n--- Phase 2: Mining Hard Negatives ---")

        X_full = df_train[self.features]

        # 2. Generate Predictions from all Scouts
        print("Generating Scout predictions on full training set...")
        preds_lgbm = scouts["lgbm"].predict(X_full)
        preds_xgb = scouts["xgb"].predict(X_full)

        # 3. Identify Hard Negatives
        # Condition: Actual Class is Negative AND (Any Scout Probability > Threshold)
        is_negative = (df_train["contact"] == 0).values

        # Union of Scout detections (Diversity Mining)
        detected_potential = (preds_lgbm > Config.MINING_THRESHOLD) | (
            preds_xgb > Config.MINING_THRESHOLD
        )

        hard_negative_mask = is_negative & detected_potential

        # Get indices
        hard_negative_indices = df_train.index[hard_negative_mask].values

        print(
            f"Mining Complete. Found {len(hard_negative_indices)} Hard Negatives out of {np.sum(is_negative)} total negatives."
        )
        print(
            f"Hard Negative Rate: {len(hard_negative_indices) / np.sum(is_negative):.4f}"
        )

        # 4. Save Cache
        print(f"Saving hard negative indices to {cache_path}...")
        save_cache(hard_negative_indices, cache_path)

        return hard_negative_indices

    def _apply_temporal_smoothing(self, df):
        """
        Applies Gaussian smoothing to the contact labels within each player pair sequence.
        """
        print("Applying Temporal Label Smoothing...")

        # Create a unique pair identifier for grouping
        # We need to ensure we don't smooth across different plays or pairs
        # pair_id = game_play + player1 + player2
        # Note: We must handle the 'G' in player 2.

        df = df.copy()
        df["_pair_id"] = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )

        # Sort by pair and step to ensure temporal order
        df.sort_values(by=["_pair_id", "step"], inplace=True)

        # Define smoothing function wrapper
        def smooth_group(x):
            return gaussian_smooth_labels(x.values, sigma=Config.LABEL_SMOOTHING_SIGMA)

        # Apply transform
        # Note: This can be memory intensive.
        df["contact_smoothed"] = df.groupby("_pair_id")["contact"].transform(
            smooth_group
        )

        # Cleanup
        df.drop(columns=["_pair_id"], inplace=True)

        return df

    def construct_anchored_dataset(
        self, df_train, hard_negative_indices, anchor_ratio=1.0
    ):
        """
        Phase 3: Anchored Dataset Construction.
        Builds the dataset for the Expert models.

        Composition:
        1. All Positives
        2. Mined Hard Negatives
        3. Random Easy Negatives (Anchors) - to prevent model collapse.

        Args:
            df_train (pd.DataFrame): Full training dataframe.
            hard_negative_indices (np.ndarray): Indices of mined hard negatives.
            anchor_ratio (float): Ratio of anchors to positives (default 1.0).

        Returns:
            tuple: (X_expert, y_expert) where y_expert is smoothed.
        """
        print("\n--- Phase 3: Constructing Anchored Expert Dataset ---")

        # 1. Apply Temporal Label Smoothing to the FULL dataset first
        # This ensures the smoothed labels respect the temporal context before we chop up the rows.
        df_smoothed = self._apply_temporal_smoothing(df_train)

        # 2. Identify Indices
        # Positives
        pos_indices = df_smoothed[df_smoothed["contact"] == 1].index.values

        # Hard Negatives (passed in)
        # Ensure intersection with current df indices (safety check)
        hard_indices = np.intersect1d(df_smoothed.index.values, hard_negative_indices)

        # Anchors (Easy Negatives)
        # Candidates are indices that are NOT positive and NOT hard negatives
        all_indices = df_smoothed.index.values
        exclude_indices = np.union1d(pos_indices, hard_indices)
        candidate_anchor_indices = np.setdiff1d(all_indices, exclude_indices)

        # Sample Anchors
        n_anchors = int(len(pos_indices) * anchor_ratio)
        # Ensure we don't sample more than available
        n_anchors = min(n_anchors, len(candidate_anchor_indices))

        rng = np.random.RandomState(Config.SEED)
        anchor_indices = rng.choice(
            candidate_anchor_indices, size=n_anchors, replace=False
        )

        print(f"Dataset Composition:")
        print(f"  Positives:      {len(pos_indices)}")
        print(f"  Hard Negatives: {len(hard_indices)}")
        print(f"  Anchors (Easy): {len(anchor_indices)}")

        # 3. Combine
        final_indices = np.concatenate([pos_indices, hard_indices, anchor_indices])

        # Shuffle
        rng.shuffle(final_indices)

        df_expert = df_smoothed.loc[final_indices]

        print(f"Final Expert Dataset Size: {len(df_expert)}")

        X_expert = df_expert[self.features]
        # Use smoothed labels for training
        y_expert = df_expert["contact_smoothed"]

        return X_expert, y_expert
