import pandas as pd
import numpy as np
from library.feature_engineering import FeatureEngineer
from library.config import FeatureConfig


class DataLoader:
    """
    Orchestrates the data loading and preparation pipeline.
    Delegates complex feature engineering and caching to the FeatureEngineer class.
    Provides interfaces for retrieving Train/Val/Test datasets and identifying feature columns.
    """

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.config = FeatureConfig()

        # Columns that are purely metadata or targets and should not be used as input features
        self.metadata_cols = {
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "contact",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "p2_merge_id",
            "nfl_player_id",
        }

    def prepare_train_val_dataset(self, load_cached=True, sample_fraction=None):
        """
        Loads and prepares the training and validation datasets.

        Args:
            load_cached (bool): If True, attempts to load from parquet cache.
            sample_fraction (float, optional): If provided (0.0 < x <= 1.0), returns a random
                                             sample of the data for debugging/quick iteration.

        Returns:
            tuple: (train_df, val_df)
        """
        # Delegate to FeatureEngineer which handles loading, merging, gating, and caching
        train_df, val_df = self.engineer.process_train_val(load_cached=load_cached)

        # Optional sampling for debugging
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            print(f"Sampling {sample_fraction:.1%} of the data for debugging...")
            train_df = train_df.sample(
                frac=sample_fraction, random_state=self.config.SEED
            ).reset_index(drop=True)
            val_df = val_df.sample(
                frac=sample_fraction, random_state=self.config.SEED
            ).reset_index(drop=True)

        return train_df, val_df

    def prepare_test_dataset(self, load_cached=True):
        """
        Loads and prepares the test dataset.

        Args:
            load_cached (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: The processed test dataframe.
        """
        # Delegate to FeatureEngineer
        test_df = self.engineer.process_test(load_cached=load_cached)
        return test_df

    def get_feature_columns(self, df):
        """
        Identifies feature columns in the dataframe by excluding known metadata and target columns.

        Args:
            df (pd.DataFrame): The dataframe containing features and metadata.

        Returns:
            list: A list of column names to be used as model features.
        """
        # Enforce Explicit Type Filtering for Model Features (Cite debug_lesson_19)
        # Select only numeric columns to prevent object/string leakage (e.g. team_p1, datetime_p2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in self.metadata_cols]
        return feature_cols

    def split_features_target(self, df, target_col="contact"):
        """
        Splits a dataframe into features (X) and target (y).

        Args:
            df (pd.DataFrame): The dataframe to split.
            target_col (str): The name of the target column.

        Returns:
            tuple: (X, y) where X is a DataFrame of features and y is a Series of the target.
        """
        feature_cols = self.get_feature_columns(df)

        X = df[feature_cols]

        if target_col in df.columns:
            y = df[target_col]
        else:
            y = None

        return X, y
