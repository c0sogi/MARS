import pandas as pd
import numpy as np
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import seed_everything


class DataPipeline:
    """
    Orchestrates data preparation for the Split-Stream Kinematic Architecture.
    Interfaces with FeatureEngineer to retrieve processed features and applies
    high-level dataset management (splitting, subsampling, formatting).
    """

    def __init__(self, config=Config):
        self.config = config
        self.fe = FeatureEngineer(config)
        seed_everything(self.config.SEED)

    def get_stream_a_data(self, split="train", load_cached=True, subsample_size=None):
        """
        Prepares data for Stream A (Interaction Model / GBDT).

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached feature files.
            subsample_size (int, optional): If set and split is 'train', downsamples the
                                          negative class to reduce dataset size while
                                          keeping all positives.

        Returns:
            pd.DataFrame: The dataset containing metadata, features, and target.
            list: List of column names to be used as features for the model.
        """
        # 1. Retrieve Feature-Engineered Data
        # FeatureEngineer handles the caching of the heavy feature generation process.
        df = self.fe.create_stream_a_features(split=split, load_cached=load_cached)

        # 2. Identify Feature Columns
        # The FE generates columns with suffixes like _0, _1, _minus_1 based on STREAM_A_FEATURES
        feature_cols = []
        for col in df.columns:
            # Check if column starts with any of the base feature names
            # and is not a metadata column
            is_feature = False
            for base_feat in self.config.STREAM_A_FEATURES:
                if col.startswith(base_feat):
                    is_feature = True
                    break

            if is_feature:
                feature_cols.append(col)

        # 3. Apply Subsampling (Train only)
        if split == "train" and subsample_size is not None and subsample_size < len(df):
            print(f"Subsampling Stream A data to ~{subsample_size} rows...")

            # Separate positives and negatives
            pos_df = df[df["contact"] == 1]
            neg_df = df[df["contact"] == 0]

            # Calculate how many negatives we can keep
            n_pos = len(pos_df)
            n_neg_target = subsample_size - n_pos

            if n_neg_target > 0:
                # Sample negatives
                neg_df_sampled = neg_df.sample(
                    n=min(len(neg_df), n_neg_target), random_state=self.config.SEED
                )

                # Recombine and shuffle
                df = pd.concat([pos_df, neg_df_sampled], axis=0)
                df = df.sample(frac=1, random_state=self.config.SEED).reset_index(
                    drop=True
                )
            else:
                # If we have more positives than the target size (unlikely), just take positives
                df = pos_df.sample(n=subsample_size, random_state=self.config.SEED)

            print(f"Subsampled data shape: {df.shape}")
            print(f"Class balance: {df['contact'].mean():.4f}")

        return df, feature_cols

    def get_stream_b_data(self, split="train", load_cached=True):
        """
        Prepares data for Stream B (Impact Model / 1D-CNN).

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached feature files.

        Returns:
            np.ndarray: X tensor of shape (N, Channels, Time).
            np.ndarray: y vector of shape (N,). None for test split.
            pd.DataFrame: Metadata corresponding to the samples.
        """
        # 1. Retrieve Feature-Engineered Data
        # FeatureEngineer returns the tensor X and the metadata DataFrame
        X, meta_df = self.fe.create_stream_b_features(
            split=split, load_cached=load_cached
        )

        # 2. Extract Target
        y = None
        if "contact" in meta_df.columns:
            y = meta_df["contact"].values.astype(np.float32)

        return X, y, meta_df

    def prepare_datasets(self, load_cached=True):
        """
        High-level wrapper to prepare all standard datasets for the pipeline.
        Useful for verifying the entire data pipeline runs correctly.
        """
        print("Preparing Stream A (Interaction) Datasets...")
        train_df_a, feats_a = self.get_stream_a_data("train", load_cached=load_cached)
        val_df_a, _ = self.get_stream_a_data("val", load_cached=load_cached)

        print("Preparing Stream B (Impact) Datasets...")
        X_train_b, y_train_b, meta_train_b = self.get_stream_b_data(
            "train", load_cached=load_cached
        )
        X_val_b, y_val_b, meta_val_b = self.get_stream_b_data(
            "val", load_cached=load_cached
        )

        return {
            "stream_a": {"train": train_df_a, "val": val_df_a, "features": feats_a},
            "stream_b": {
                "train": (X_train_b, y_train_b),
                "val": (X_val_b, y_val_b),
                "meta_train": meta_train_b,
                "meta_val": meta_val_b,
            },
        }
