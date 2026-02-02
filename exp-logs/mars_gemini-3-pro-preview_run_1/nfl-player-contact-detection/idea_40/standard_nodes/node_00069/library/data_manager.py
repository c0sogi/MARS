import os
import numpy as np
import pandas as pd
import gc
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer


class DataManager:
    """
    Manages data loading, dataset construction, and hard negative mining
    for the Relative-Velocity-Aligned Anchored-Mining Ensemble.
    """

    def __init__(self, config=Config):
        self.config = config
        self.fe = FeatureEngineer(config)

        # Columns to exclude from X (features)
        self.metadata_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "gating_active",
            "datetime",
        ]

    def get_train_data(self, load_cached_data=True):
        """
        Loads training features using the FeatureEngineer.
        """
        return self.fe.process_train(load_cached_data=load_cached_data)

    def get_val_data(self, load_cached_data=True):
        """
        Loads validation features using the FeatureEngineer.
        """
        return self.fe.process_val(load_cached_data=load_cached_data)

    def get_test_data(self, load_cached_data=True):
        """
        Loads test features using the FeatureEngineer.
        """
        return self.fe.process_test(load_cached_data=load_cached_data)

    def build_scout_dataset(self, df_train):
        """
        Constructs a balanced (1:1) dataset for training Scout models.

        Args:
            df_train (pd.DataFrame): The full training dataframe (gated survivors).

        Returns:
            pd.DataFrame: The balanced dataset.
        """
        seed_everything(self.config.SEED)

        # Separate positives and negatives
        positives = df_train[df_train["contact"] == 1]
        negatives = df_train[df_train["contact"] == 0]

        n_pos = len(positives)

        # Sample negatives to match positives (1:1)
        # If we have fewer negatives than positives (unlikely), take all negatives
        n_neg = min(len(negatives), n_pos)

        negatives_sampled = negatives.sample(n=n_neg, random_state=self.config.SEED)

        # Combine and shuffle
        df_scout = pd.concat([positives, negatives_sampled], axis=0)
        df_scout = df_scout.sample(frac=1, random_state=self.config.SEED).reset_index(
            drop=True
        )

        return df_scout

    def mine_hard_negatives(self, df_train, scout_models, load_cached_indices=True):
        """
        Mines hard negatives using trained Scout models.
        Hard Negatives are negatives where P(Contact) > Threshold for ANY scout.

        Args:
            df_train (pd.DataFrame): The full training dataframe.
            scout_models (list): List of trained model wrappers (must have .predict()).
            load_cached_indices (bool): Whether to load indices from cache.

        Returns:
            np.array: Array of indices (from df_train) corresponding to hard negatives.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, "hard_negative_indices.npy")

        if load_cached_indices and os.path.exists(cache_path):
            print(f"Loading cached hard negative indices from {cache_path}")
            return np.load(cache_path)

        print("Mining Hard Negatives with Scout models...")

        # Prepare features for inference
        X_train, _ = self.get_X_y(df_train)

        # Get predictions from all scouts
        all_preds = []
        for model in scout_models:
            preds = model.predict(X_train)
            all_preds.append(preds)

        # Calculate max probability across scouts (Union strategy)
        # If any scout says prob > threshold, it's a candidate
        max_preds = np.max(np.vstack(all_preds), axis=0)

        # Identify Hard Negatives
        # Criteria: Actual Class is 0 AND Max Predicted Prob > Threshold
        is_negative = (df_train["contact"] == 0).values
        is_hard = max_preds > self.config.HARD_NEGATIVE_THRESHOLD

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = df_train.index[hard_negative_mask].values

        print(f"Found {len(hard_negative_indices)} Hard Negatives.")

        # Cache the indices
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, hard_negative_indices)
        print(f"Hard negative indices saved to {cache_path}")

        return hard_negative_indices

    def build_expert_dataset(self, df_train, hard_negative_indices):
        """
        Constructs the Expert Dataset using:
        1. All Positives
        2. Mined Hard Negatives
        3. Random Anchors (Easy Negatives)

        Args:
            df_train (pd.DataFrame): The full training dataframe.
            hard_negative_indices (np.array): Indices of mined hard negatives.

        Returns:
            pd.DataFrame: The compiled Expert dataset.
        """
        seed_everything(self.config.SEED)

        # 1. All Positives
        positives = df_train[df_train["contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid
        valid_indices = np.intersect1d(hard_negative_indices, df_train.index)
        hard_negatives = df_train.loc[valid_indices]

        # 3. Random Anchors
        # Pool of all negatives
        all_negatives = df_train[df_train["contact"] == 0]

        # Determine number of anchors based on ratio to positives
        n_anchors = int(len(positives) * self.config.ANCHOR_RATIO)

        # Sample anchors
        # We sample from ALL negatives to ensure we get a representative distribution of "easy" cases.
        # Overlap with hard negatives is acceptable (and statistically valid for anchors),
        # but we can deduplicate later if we strictly want "Easy" vs "Hard".
        # The prompt implies "Random Easy Negatives (Anchors)" vs "Hard Negatives".
        # A simple random sample provides the anchors.
        anchors = all_negatives.sample(
            n=min(len(all_negatives), n_anchors), random_state=self.config.SEED
        )

        # Combine
        # Use index to handle potential overlaps if we want unique rows,
        # though duplicates (re-weighting) for hard negatives that are also anchors is not necessarily bad.
        # However, to keep it clean, we'll concat and drop duplicates by index.
        df_expert = pd.concat([positives, hard_negatives, anchors], axis=0)

        # Remove duplicate rows (based on index) to prevent data leakage/double counting
        df_expert = df_expert[~df_expert.index.duplicated(keep="first")]

        # Shuffle
        df_expert = df_expert.sample(frac=1, random_state=self.config.SEED).reset_index(
            drop=True
        )

        print(f"Expert Dataset constructed: {len(df_expert)} rows.")
        print(f"  Positives: {len(positives)}")
        print(f"  Hard Negatives (Unique): {len(hard_negatives)}")
        print(
            f"  Anchors (Unique contribution): {len(df_expert) - len(positives) - len(hard_negatives)}"
        )

        return df_expert

    def get_X_y(self, df):
        """
        Splits DataFrame into Features (X) and Target (y).
        Removes metadata columns.

        Args:
            df (pd.DataFrame): Dataframe containing features and target.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        # Identify feature columns (all columns that are not metadata)
        feature_cols = [c for c in df.columns if c not in self.metadata_cols]

        X = df[feature_cols]
        y = df["contact"]

        return X, y
