import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import save_npy, load_npy


class DataManager:
    """
    Manages data loading, dataset construction for Scouts and Experts,
    and handling of hard negative indices.
    """

    def __init__(self):
        self.fe = FeatureEngineer()
        self.metadata_cols = [
            "contact_id",
            "game_play",
            "game_key",
            "play_id",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
        ]

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata and target columns.
        """
        return [c for c in df.columns if c not in self.metadata_cols]

    def get_train_features(self, load_cached_data=True):
        """
        Returns the full training feature set (post-gating).
        """
        return self.fe.create_train_features(load_cached_data=load_cached_data)

    def get_val_features(self, load_cached_data=True):
        """
        Returns the validation feature set.
        """
        return self.fe.create_val_features(load_cached_data=load_cached_data)

    def get_test_features(self, load_cached_data=True):
        """
        Returns the test feature set.
        """
        return self.fe.create_test_features(load_cached_data=load_cached_data)

    def get_scout_dataset(self, df_train):
        """
        Constructs a balanced dataset for Scout training.
        Strategy: All Positives + Equal number of Random Negatives.

        Args:
            df_train (pd.DataFrame): Full training dataframe containing features and 'contact' target.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        print("Constructing Scout Dataset (Balanced 1:1)...")

        # Separate classes
        positives = df_train[df_train["contact"] == 1]
        negatives = df_train[df_train["contact"] == 0]

        n_pos = len(positives)
        n_neg = len(negatives)

        # Downsample negatives
        if n_neg > n_pos:
            negatives = negatives.sample(n=n_pos, random_state=Config.SEED)

        # Combine and shuffle
        df_scout = pd.concat([positives, negatives], axis=0)
        df_scout = shuffle(df_scout, random_state=Config.SEED).reset_index(drop=True)

        # Split X and y
        feature_cols = self._get_feature_columns(df_scout)
        X = df_scout[feature_cols]
        y = df_scout["contact"]

        print(
            f"Scout Dataset Size: {len(X)} (Pos: {len(positives)}, Neg: {len(negatives)})"
        )
        return X, y

    def get_expert_dataset(self, df_train, hard_negative_indices):
        """
        Constructs the Expert dataset.
        Strategy:
            1. All Positives
            2. Mined Hard Negatives (Union of Scout failures)
            3. Buffer of Random Negatives (1:1 ratio with Positives)

        Args:
            df_train (pd.DataFrame): Full training dataframe.
            hard_negative_indices (np.array): Indices of rows in df_train identified as hard negatives.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        print(
            f"Constructing Expert Dataset with {len(hard_negative_indices)} hard negatives..."
        )

        # 1. All Positives
        positives = df_train[df_train["contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid
        valid_indices = np.intersect1d(hard_negative_indices, df_train.index)
        df_hard_neg = df_train.loc[valid_indices]

        # Ensure they are actually negatives (sanity check)
        df_hard_neg = df_hard_neg[df_hard_neg["contact"] == 0]

        # 3. Random Buffer
        # We want some easy negatives to maintain distribution stability.
        # Let's take a buffer equal to the number of positives.
        # We must exclude the hard negatives from this pool to avoid duplication.
        n_buffer = len(positives)

        # Get all negatives
        all_neg_indices = df_train[df_train["contact"] == 0].index

        # Exclude hard negative indices
        available_easy_indices = np.setdiff1d(all_neg_indices, valid_indices)

        # Sample buffer
        if len(available_easy_indices) > n_buffer:
            np.random.seed(Config.SEED)
            buffer_indices = np.random.choice(
                available_easy_indices, size=n_buffer, replace=False
            )
            df_buffer = df_train.loc[buffer_indices]
        else:
            df_buffer = df_train.loc[available_easy_indices]

        # Combine
        df_expert = pd.concat([positives, df_hard_neg, df_buffer], axis=0)
        df_expert = shuffle(df_expert, random_state=Config.SEED).reset_index(drop=True)

        # Split X and y
        feature_cols = self._get_feature_columns(df_expert)
        X = df_expert[feature_cols]
        y = df_expert["contact"]

        print(f"Expert Dataset Size: {len(X)}")
        print(f"  Positives: {len(positives)}")
        print(f"  Hard Negatives: {len(df_hard_neg)}")
        print(f"  Random Buffer: {len(df_buffer)}")

        return X, y

    def save_hard_negative_indices(self, indices):
        """
        Saves the indices of identified hard negatives to disk.
        """
        save_npy(indices, Config.CACHE_HARD_NEGATIVE_INDICES)
        print(
            f"Saved {len(indices)} hard negative indices to {Config.CACHE_HARD_NEGATIVE_INDICES}"
        )

    def load_hard_negative_indices(self):
        """
        Loads hard negative indices from disk.
        """
        if os.path.exists(Config.CACHE_HARD_NEGATIVE_INDICES):
            return load_npy(Config.CACHE_HARD_NEGATIVE_INDICES)
        return np.array([])
