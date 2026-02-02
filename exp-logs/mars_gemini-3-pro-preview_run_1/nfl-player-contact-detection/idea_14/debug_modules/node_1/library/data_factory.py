import os
import pandas as pd
from typing import Optional, Tuple, Union
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    GATING_DIST,
    FeatureConfig,
    SEED,
)
from library.feature_engineering import FeatureEngineer
from library.utils import set_seed


class DataFactory:
    """
    Orchestrates data loading, preprocessing, and feature engineering.
    Manages the pipeline from raw CSVs to the final feature matrix used for training.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config if config is not None else FeatureConfig()
        self.engineer = FeatureEngineer(self.config)
        set_seed(SEED)

    def load_raw_data(
        self, mode: str = "train", sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads raw metadata and tracking data from CSVs.

        Args:
            mode: 'train', 'val', or 'test'.
            sample_size: If provided, samples the metadata to this number of rows.

        Returns:
            Tuple of (metadata_df, tracking_df)
        """
        if mode == "train":
            meta_path = TRAIN_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH
        elif mode == "val":
            meta_path = VAL_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH
        elif mode == "test":
            meta_path = TEST_METADATA_PATH
            track_path = TEST_TRACKING_PATH
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Load Metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Apply sampling if requested
        if sample_size is not None and sample_size < len(df_meta):
            df_meta = df_meta.sample(n=sample_size, random_state=SEED).reset_index(
                drop=True
            )

        # Load Tracking
        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Tracking file not found: {track_path}")

        df_track = pd.read_csv(track_path)

        return df_meta, df_track

    def apply_geometric_gating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataset to remove easy negatives (Player-Player > GATING_DIST).
        Keeps all Player-Ground interactions.

        This logic is critical for reducing class imbalance and dataset size.
        """
        # Ensure necessary columns exist
        if "distance" not in df.columns or "is_ground" not in df.columns:
            return df

        # Logic: Keep if distance is small OR if it is a ground contact
        # Note: Ground contacts usually have distance=0 set during feature engineering,
        # but explicit check is safer.
        mask = (df["distance"] < GATING_DIST) | (df["is_ground"] == 1)
        return df[mask].copy()

    def get_processed_dataset(
        self,
        mode: str = "train",
        load_cached_data: bool = True,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates the feature matrix for the specified mode.

        Args:
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cached parquet files.
            sample_size: If set, generates features for a subset of data.
                         Note: When sampling, cache usage is modified to avoid collisions.

        Returns:
            pd.DataFrame: Feature matrix ready for model training/inference.
        """
        # Case 1: Sampling requested.
        # We must load raw data, sample it, and then generate features.
        # We modify the mode string to ensure we don't overwrite the full dataset cache.
        if sample_size is not None:
            effective_mode = f"{mode}_sample_{sample_size}"

            # Load and sample raw data
            df_meta, df_track = self.load_raw_data(mode, sample_size)

            # Generate features
            # FeatureEngineer handles the heavy lifting
            df_features = self.engineer.generate_features(
                metadata_input=df_meta,
                tracking_input=df_track,
                mode=effective_mode,
                load_cached_data=load_cached_data,
            )
            return df_features

        # Case 2: Full Dataset.
        # We pass file paths to FeatureEngineer to allow it to handle IO efficiently.
        if mode == "train":
            meta_input = TRAIN_METADATA_PATH
            track_input = TRAIN_TRACKING_PATH
        elif mode == "val":
            meta_input = VAL_METADATA_PATH
            track_input = TRAIN_TRACKING_PATH
        elif mode == "test":
            meta_input = TEST_METADATA_PATH
            track_input = TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        df_features = self.engineer.generate_features(
            metadata_input=meta_input,
            tracking_input=track_input,
            mode=mode,
            load_cached_data=load_cached_data,
        )

        return df_features
