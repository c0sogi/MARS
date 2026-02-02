import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger
from library.feature_engineering import process_dataset


class DataManager:
    """
    Manages data loading, preprocessing, and dataset construction for the
    Quadratic-Gated Spectral-Mining Ensemble.
    """

    def __init__(self):
        self.logger = setup_logger("data_manager")
        # Define the feature columns used for training/inference
        self.feature_cols = [
            "distance",
            "rel_speed",
            "rel_accel",
            "ke_p1",
            "ke_p2",
            "spectral_energy",
            "gating_min_dist",
        ]

    def load_train_features(self, load_cached=True):
        """
        Loads the training data with full spectral features and quadratic gating applied.
        Ensures a reset index for consistent row addressing during mining.
        """
        self.logger.info("Loading Train features...")
        df = process_dataset(
            Config.TRAIN_METADATA_PATH,
            Config.TRACKING_PATH_TRAIN,
            dataset_type="train",
            load_cached_data=load_cached,
        )
        # Ensure index is 0..N regardless of whether it came from cache or fresh processing
        return df.reset_index(drop=True)

    def load_val_features(self, load_cached=True):
        """
        Loads the validation data.
        """
        self.logger.info("Loading Validation features...")
        df = process_dataset(
            Config.VAL_METADATA_PATH,
            Config.TRACKING_PATH_TRAIN,  # Validation plays are in the train tracking file
            dataset_type="val",
            load_cached_data=load_cached,
        )
        return df.reset_index(drop=True)

    def load_test_features(self, load_cached=True):
        """
        Loads the test data.
        """
        self.logger.info("Loading Test features...")
        df = process_dataset(
            Config.TEST_METADATA_PATH,
            Config.TRACKING_PATH_TEST,
            dataset_type="test",
            load_cached_data=load_cached,
        )
        return df.reset_index(drop=True)

    def get_scout_dataset(self, df_train):
        """
        Constructs a balanced dataset (1:1 Positive:Negative) for training Scout models.
        Used in Phase 1 of the curriculum.
        """
        self.logger.info("Constructing Balanced Scout Dataset...")

        # Separate Positives and Negatives
        pos_mask = df_train["contact"] == 1
        df_pos = df_train[pos_mask]
        df_neg = df_train[~pos_mask]

        # Calculate number of negatives to sample
        n_pos = len(df_pos)
        ratio = Config.TRAINING.get("NEGATIVE_SAMPLING_RATIO", 1.0)
        n_neg = int(n_pos * ratio)

        # Sample Negatives
        if n_neg > len(df_neg):
            n_neg = len(df_neg)

        df_neg_sample = df_neg.sample(n=n_neg, random_state=Config.SEED)

        # Combine and Shuffle
        df_scout = pd.concat([df_pos, df_neg_sample], axis=0)
        df_scout = df_scout.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        X = df_scout[self.feature_cols].values
        y = df_scout["contact"].values

        self.logger.info(
            f"Scout Data: {len(df_scout)} rows (Pos: {len(df_pos)}, Neg: {len(df_neg_sample)})"
        )
        return X, y

    def get_expert_dataset(self, df_train, hard_negative_indices):
        """
        Constructs the Expert Dataset for Phase 3.
        Composition: All Positives + Mined Hard Negatives + Random Negative Buffer.

        Args:
            df_train: The full training dataframe (gated).
            hard_negative_indices: List/Array of indices in df_train identified as hard negatives.
        """
        self.logger.info(
            f"Constructing Expert Dataset with {len(hard_negative_indices)} hard negatives..."
        )

        # 1. All Positives
        pos_mask = df_train["contact"] == 1
        df_pos = df_train[pos_mask]

        # 2. Hard Negatives
        # Retrieve rows by index. Ensure they are actually negatives (safety check).
        df_hard = df_train.loc[hard_negative_indices]
        df_hard = df_hard[df_hard["contact"] == 0]

        # 3. Buffer Random Negatives
        # We sample from negatives that are NOT in the hard negative set
        neg_mask = df_train["contact"] == 0
        all_neg_indices = df_train[neg_mask].index

        # Exclude hard negatives from the buffer pool
        hard_indices_set = set(df_hard.index)
        buffer_pool = [i for i in all_neg_indices if i not in hard_indices_set]

        # Determine buffer size
        n_pos = len(df_pos)
        n_buffer = int(n_pos * Config.TRAINING.get("EXPERT_NEGATIVE_BUFFER", 0.5))

        if n_buffer > len(buffer_pool):
            n_buffer = len(buffer_pool)

        # Sample buffer using numpy for efficiency
        rng = np.random.RandomState(Config.SEED)
        buffer_indices = rng.choice(buffer_pool, size=n_buffer, replace=False)
        df_buffer = df_train.loc[buffer_indices]

        # Combine All
        df_expert = pd.concat([df_pos, df_hard, df_buffer], axis=0)
        df_expert = df_expert.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        X = df_expert[self.feature_cols].values
        y = df_expert["contact"].values

        self.logger.info(
            f"Expert Data: {len(df_expert)} rows "
            f"(Pos: {len(df_pos)}, HardNeg: {len(df_hard)}, Buffer: {len(df_buffer)})"
        )
        return X, y

    def get_validation_set(self, df_val):
        """
        Returns X, y for the validation set.
        """
        X = df_val[self.feature_cols].values
        y = df_val["contact"].values
        return X, y

    def get_test_set(self, df_test):
        """
        Returns X, contact_ids for the test set.
        """
        X = df_test[self.feature_cols].values
        ids = df_test["contact_id"].values
        return X, ids

    def get_feature_matrix(self, df):
        """
        Helper to extract just the feature matrix X from any dataframe.
        """
        return df[self.feature_cols].values
