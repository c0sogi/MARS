import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logging, seed_everything, save_numpy, load_numpy
from library.feature_extractor import FeatureExtractor


class DataManager:
    def __init__(self):
        self.config = Config
        self.extractor = FeatureExtractor()
        setup_logging()
        seed_everything(self.config.SEED)

    def load_train_features(self, load_cached_data=True):
        """
        Generates or loads the full gated training feature set.
        """
        return self.extractor.generate_features(
            metadata_path=self.config.TRAIN_METADATA_PATH,
            tracking_path=self.config.TRAIN_TRACKING_PATH,
            mode="train",
            load_cached_data=load_cached_data,
        )

    def load_val_features(self, load_cached_data=True):
        """
        Generates or loads the validation feature set.
        """
        return self.extractor.generate_features(
            metadata_path=self.config.VAL_METADATA_PATH,
            tracking_path=self.config.TRAIN_TRACKING_PATH,
            mode="val",
            load_cached_data=load_cached_data,
        )

    def load_test_features(self, load_cached_data=True):
        """
        Generates or loads the test feature set (no gating).
        """
        return self.extractor.generate_features(
            metadata_path=self.config.TEST_METADATA_PATH,
            tracking_path=self.config.TEST_TRACKING_PATH,
            mode="test",
            load_cached_data=load_cached_data,
        )

    def get_scout_dataset(self, df_features):
        """
        Constructs a balanced dataset for Scout training.
        Positives + Random Sample of Negatives.
        """
        print("Constructing Scout Dataset...")

        # Separate classes
        pos_mask = df_features["contact"] == 1
        neg_mask = df_features["contact"] == 0

        df_pos = df_features[pos_mask]
        df_neg = df_features[neg_mask]

        n_pos = len(df_pos)
        n_neg_target = int(n_pos * self.config.MINING["NEG_POS_RATIO"])

        # Sample negatives
        if len(df_neg) > n_neg_target:
            df_neg_sampled = df_neg.sample(
                n=n_neg_target, random_state=self.config.SEED
            )
        else:
            df_neg_sampled = df_neg

        # Combine and shuffle
        df_scout = (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1.0, random_state=self.config.SEED)
            .reset_index(drop=True)
        )

        X = df_scout.drop(columns=["contact"])
        y = df_scout["contact"]

        print(
            f"Scout Dataset: {len(X)} samples (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
        )
        return X, y

    def get_expert_dataset(self, df_features, hard_negative_indices):
        """
        Constructs the Expert dataset.
        Positives + Hard Negatives (from indices) + Random Buffer Negatives.
        """
        print(
            f"Constructing Expert Dataset with {len(hard_negative_indices)} hard negatives..."
        )

        # 1. Positives
        df_pos = df_features[df_features["contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid
        valid_indices = [
            idx for idx in hard_negative_indices if idx in df_features.index
        ]
        df_hard_neg = df_features.loc[valid_indices]

        # Verify they are actually negatives (sanity check)
        # In case a positive was flagged as a hard negative (unlikely but possible if label noise)
        df_hard_neg = df_hard_neg[df_hard_neg["contact"] == 0]

        # 3. Buffer Negatives
        # Pool of negatives excluding hard negatives
        neg_mask = (df_features["contact"] == 0) & (
            ~df_features.index.isin(valid_indices)
        )
        df_neg_pool = df_features[neg_mask]

        n_pos = len(df_pos)
        n_buffer = int(n_pos * self.config.MINING["BUFFER_RATIO"])

        if len(df_neg_pool) > n_buffer:
            df_buffer = df_neg_pool.sample(n=n_buffer, random_state=self.config.SEED)
        else:
            df_buffer = df_neg_pool

        # Combine
        df_expert = pd.concat([df_pos, df_hard_neg, df_buffer], axis=0)

        # Shuffle
        df_expert = df_expert.sample(
            frac=1.0, random_state=self.config.SEED
        ).reset_index(drop=True)

        X = df_expert.drop(columns=["contact"])
        y = df_expert["contact"]

        print(f"Expert Dataset: {len(X)} samples")
        print(f"  Positives: {len(df_pos)}")
        print(f"  Hard Negatives: {len(df_hard_neg)}")
        print(f"  Buffer Negatives: {len(df_buffer)}")

        return X, y

    def save_hard_negatives(self, indices):
        """
        Saves hard negative indices to disk.
        """
        path = self.config.get_cache_path("hard_negatives")
        print(f"Saving {len(indices)} hard negative indices to {path}")
        save_numpy(np.array(indices), path)

    def load_hard_negatives(self):
        """
        Loads hard negative indices from disk.
        """
        path = self.config.get_cache_path("hard_negatives")
        if os.path.exists(path):
            return load_numpy(path)
        return np.array([])
