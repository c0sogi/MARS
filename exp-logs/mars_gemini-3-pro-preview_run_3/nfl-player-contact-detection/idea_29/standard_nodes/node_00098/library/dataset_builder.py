import pandas as pd
import numpy as np
from library.config import Config
from library.feature_engineering import FeatureEngineer


class DatasetBuilder:
    """
    Orchestrates the data preparation pipeline with caching and validation.
    Manages the generation of features for Stream A (Player-Player) and Stream B (Player-Ground),
    ensuring schema consistency and data integrity.
    """

    def __init__(self, split: str, load_cached_data: bool = True):
        """
        Initialize the DatasetBuilder.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to use cached data if available.
        """
        self.split = split
        self.load_cached_data = load_cached_data
        self.feature_engineer = FeatureEngineer(split, load_cached_data)

    def build_dataset(self, stream: str):
        """
        Builds the dataset for a specific stream.
        Checks cache via FeatureEngineer, generates if missing, and validates schema.

        Args:
            stream (str): 'A' for Interaction (Player-Player), 'B' for Impact (Player-Ground).

        Returns:
            tuple: (X, ids, y)
                X (pd.DataFrame): Feature matrix.
                ids (np.array): Contact IDs.
                y (np.array): Labels.
        """
        if stream == "A":
            X, ids, y = self.feature_engineer.process_stream_a()
        elif stream == "B":
            X, ids, y = self.feature_engineer.process_stream_b()
        else:
            raise ValueError(f"Invalid stream: {stream}. Must be 'A' or 'B'.")

        # Validate Schema if data exists
        if not X.empty:
            self.validate_schema(X, stream)

        return X, ids, y

    def validate_schema(self, X: pd.DataFrame, stream: str):
        """
        Validates that the feature matrix contains all expected columns and they are not zero-filled.
        Raises RuntimeError if validation fails.

        Args:
            X (pd.DataFrame): The feature matrix to validate.
            stream (str): The stream identifier ('A' or 'B').
        """
        # 1. Determine expected base features from Config
        if stream == "A":
            features_map = Config.STREAM_A_FEATURES
            base_features = (
                features_map["relational"]
                + features_map["visual"]
                + features_map["cross_modal"]
                + features_map["energy"]
            )
        elif stream == "B":
            features_map = Config.STREAM_B_FEATURES
            base_features = features_map["field_centric"] + features_map["ego_centric"]
        else:
            raise ValueError(f"Unknown stream: {stream}")

        # 2. Reconstruct expected lagged columns based on Config.LAG_SCHEDULE
        expected_columns = []
        for lag in Config.LAG_SCHEDULE:
            if lag == 0:
                expected_columns.extend(base_features)
                continue

            # Replicate naming logic from FeatureEngineer
            suffix = f"_lag_{lag}" if lag < 0 else f"_lead_{lag}"
            suffix = suffix.replace("-", "m")

            for col in base_features:
                expected_columns.append(f"{col}{suffix}")

        # 3. Check for missing columns
        missing_cols = [col for col in expected_columns if col not in X.columns]
        if missing_cols:
            raise RuntimeError(
                f"Stream {stream} Schema Validation Failed. Missing columns: {missing_cols}"
            )

        # 4. Check for zero-filled columns (indicates pipeline failure)
        # We skip this check for very small datasets (e.g., < 10 rows) to avoid false positives during debugging
        if len(X) > 10:
            zero_filled_cols = []
            for col in expected_columns:
                if (X[col] == 0).all():
                    zero_filled_cols.append(col)

            if zero_filled_cols:
                raise RuntimeError(
                    f"Stream {stream} Schema Validation Failed. Zero-filled columns detected: {zero_filled_cols}"
                )

        print(
            f"Stream {stream} Schema Validation Passed. {len(expected_columns)} features verified."
        )

    @staticmethod
    def split_streams(df: pd.DataFrame):
        """
        Separates a DataFrame into Interaction (Player-Player) and Impact (Player-Ground) subsets.
        Can handle DataFrames with 'nfl_player_id_2' column or 'contact_id' column.

        Args:
            df (pd.DataFrame): DataFrame to split.

        Returns:
            tuple: (df_stream_a, df_stream_b)
                df_stream_a: Subset where player2 != 'G'
                df_stream_b: Subset where player2 == 'G'
        """
        if "nfl_player_id_2" in df.columns:
            mask_ground = df["nfl_player_id_2"] == "G"
        elif "contact_id" in df.columns:
            # Extract player 2 from contact_id (format: game_play_step_p1_p2)
            # We assume the last segment is player 2
            p2_extracted = (
                df["contact_id"].astype(str).apply(lambda x: x.split("_")[-1])
            )
            mask_ground = p2_extracted == "G"
        else:
            raise ValueError(
                "DataFrame must contain 'nfl_player_id_2' or 'contact_id' to split streams."
            )

        df_b = df[mask_ground].copy()
        df_a = df[~mask_ground].copy()

        return df_a, df_b
