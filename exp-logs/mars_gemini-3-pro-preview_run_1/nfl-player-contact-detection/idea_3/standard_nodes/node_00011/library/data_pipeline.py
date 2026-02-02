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

    def get_unified_data(self, split="train", load_cached=True, subsample_size=None):
        """
        Prepares data for the Unified Model.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached feature files.
            subsample_size (int, optional): If set and split is 'train', downsamples the
                                          negative class.

        Returns:
            pd.DataFrame: The dataset containing metadata, features, and target.
            list: List of column names to be used as features for the model.
        """
        # 1. Retrieve Feature-Engineered Data
        df = self.fe.create_unified_features(split=split, load_cached=load_cached)

        # 2. Identify Feature Columns
        feature_cols = []
        for col in df.columns:
            is_feature = False
            for base_feat in self.config.STREAM_A_FEATURES:
                if col.startswith(base_feat):
                    is_feature = True
                    break
            if is_feature:
                feature_cols.append(col)

        # 3. Apply Subsampling (Train only)
        if split == "train" and subsample_size is not None and subsample_size < len(df):
            print(f"Subsampling Unified data to ~{subsample_size} rows...")

            pos_df = df[df["contact"] == 1]
            neg_df = df[df["contact"] == 0]

            n_pos = len(pos_df)
            n_neg_target = subsample_size - n_pos

            if n_neg_target > 0:
                neg_df_sampled = neg_df.sample(
                    n=min(len(neg_df), n_neg_target), random_state=self.config.SEED
                )
                df = pd.concat([pos_df, neg_df_sampled], axis=0)
                df = df.sample(frac=1, random_state=self.config.SEED).reset_index(
                    drop=True
                )
            else:
                df = pos_df.sample(n=subsample_size, random_state=self.config.SEED)

            print(f"Subsampled data shape: {df.shape}")
            print(f"Class balance: {df['contact'].mean():.4f}")

        return df, feature_cols
