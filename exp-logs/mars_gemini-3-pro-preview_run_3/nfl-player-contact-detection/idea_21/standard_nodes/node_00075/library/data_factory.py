import pandas as pd
import numpy as np
import os
from library.config import Config
from library.features import FeatureGenerator
from library.utils import set_seed


class DataLoader:
    """
    Handles loading of metadata and raw datasets.
    Acts as a wrapper around specific data loading logic, primarily used
    to initialize the environment for StreamBuilder.
    """

    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'.
        """
        self.mode = mode
        self.metadata_path = {
            "train": Config.TRAIN_META_PATH,
            "validation": Config.VAL_META_PATH,
            "test": Config.TEST_META_PATH,
        }[mode]

    def load_metadata(self):
        """Loads the metadata CSV for the current mode."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        return pd.read_csv(self.metadata_path)


class StreamBuilder:
    """
    Constructs the specific datasets (Stream A and Stream B) for training and inference.
    Handles feature generation (via FeatureGenerator) and undersampling strategies.
    """

    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'.
        """
        self.mode = mode
        self.feature_gen = FeatureGenerator(mode=mode)
        self.loader = DataLoader(mode=mode)

    def undersample_negatives(self, X, y, ids):
        """
        Applies Targeted Majority Undersampling.
        Retains 100% of positive samples and subsamples negative samples
        to achieve a specific ratio (Config.NEGATIVE_SAMPLE_RATIO).

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target labels.
            ids (np.array): Contact IDs.

        Returns:
            tuple: (X_sampled, y_sampled, ids_sampled)
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate number of negatives to keep
        # Ratio is Neg:Pos. e.g. 10:1
        n_neg_keep = int(n_pos * Config.NEGATIVE_SAMPLE_RATIO)

        # If we have fewer negatives than requested, keep all
        if n_neg_keep > n_neg:
            n_neg_keep = n_neg

        # Randomly sample negatives
        # Using numpy's random generator for consistency
        rng = np.random.default_rng(Config.SEED)
        neg_sampled = rng.choice(neg_indices, size=n_neg_keep, replace=False)

        # Combine indices
        keep_indices = np.concatenate([pos_indices, neg_sampled])

        # Shuffle the combined indices to mix classes
        rng.shuffle(keep_indices)

        # Filter data
        X_sampled = X.iloc[keep_indices].copy().reset_index(drop=True)
        y_sampled = y[keep_indices]
        ids_sampled = ids[keep_indices]

        print(f"Undersampling Complete ({self.mode}):")
        print(f"  Original: {len(y)} (Pos: {n_pos}, Neg: {n_neg})")
        print(f"  Sampled:  {len(y_sampled)} (Pos: {n_pos}, Neg: {len(neg_sampled)})")

        return X_sampled, y_sampled, ids_sampled

    def build_interaction_set(self, load_cached=True):
        """
        Builds the dataset for Stream A (Player-Player Interaction).
        Applies undersampling if in 'train' mode.

        Args:
            load_cached (bool): Whether to attempt loading features from cache.

        Returns:
            X (pd.DataFrame), y (np.array), ids (np.array)
        """
        # Generate/Load Features via FeatureGenerator
        X, y, ids = self.feature_gen.generate_stream_a(load_cached=load_cached)

        # Apply Undersampling only for training
        if self.mode == "train":
            X, y, ids = self.undersample_negatives(X, y, ids)

        return X, y, ids

    def build_impact_set(self, load_cached=True):
        """
        Builds the dataset for Stream B (Player-Ground Impact).
        Applies undersampling if in 'train' mode.

        Args:
            load_cached (bool): Whether to attempt loading features from cache.

        Returns:
            X (pd.DataFrame), y (np.array), ids (np.array)
        """
        # Generate/Load Features via FeatureGenerator
        X, y, ids = self.feature_gen.generate_stream_b(load_cached=load_cached)

        # Apply Undersampling only for training
        if self.mode == "train":
            X, y, ids = self.undersample_negatives(X, y, ids)

        return X, y, ids
