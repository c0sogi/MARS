import os
import pandas as pd
import numpy as np
from library import config


class FeaturePipeline:
    """
    A feature engineering pipeline that transforms raw data into
    model-ready features using geometric calculations and reverse one-hot encoding.
    """

    def __init__(self):
        pass

    def _reverse_one_hot(self, df):
        """
        Condenses sparse binary columns (Soil_Type, Wilderness_Area) into dense integer indices.
        Crucially sorts column names numerically to preserve ordinal semantics where applicable.
        """
        # 1. Process Soil Types
        soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
        # Sort numerically: Soil_Type1, Soil_Type2, ..., Soil_Type10
        soil_cols.sort(key=lambda x: int(x.replace("Soil_Type", "")))

        if soil_cols:
            # argmax finds the index of the '1' in the one-hot vector
            # We use int8 to minimize memory footprint
            df["Soil_Type_Index"] = np.argmax(df[soil_cols].values, axis=1).astype(
                np.int8
            )
            # Drop original columns to save memory
            df.drop(columns=soil_cols, inplace=True)

        # 2. Process Wilderness Areas
        wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
        wild_cols.sort(key=lambda x: int(x.replace("Wilderness_Area", "")))

        if wild_cols:
            df["Wilderness_Area_Index"] = np.argmax(
                df[wild_cols].values, axis=1
            ).astype(np.int8)
            df.drop(columns=wild_cols, inplace=True)

        return df

    def _add_geometric_features(self, df):
        """
        Adds physics-informed geometric features to the dataframe.
        """
        # Euclidean Distance to Hydrology
        # d = sqrt(h_dist^2 + v_dist^2)
        h_dist = df["Horizontal_Distance_To_Hydrology"].astype(np.float32)
        v_dist = df["Vertical_Distance_To_Hydrology"].astype(np.float32)
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            h_dist**2 + v_dist**2
        ).astype(np.float32)

        # Aspect Transformations (Cyclic Encoding)
        # Convert degrees to radians for sin/cos
        aspect_rad = np.deg2rad(df["Aspect"].astype(np.float32))
        df["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
        df["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)

        # Relative Elevation (Hydrology)
        # Elevation at the point of hydrology = Elevation - Vertical_Dist
        df["Elevation_Hydrology"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

        return df

    def fit_transform(self, df):
        """
        Fits the pipeline on the training data and transforms it.
        """
        print("Feature Engineering: Fitting and Transforming Train Set...")

        # 1. Reverse One-Hot (Independent)
        df = self._reverse_one_hot(df)

        # 2. Geometric Features (Independent)
        df = self._add_geometric_features(df)

        return df

    def transform(self, df):
        """
        Transforms the test data using the pipeline fitted on training data.
        """
        print("Feature Engineering: Transforming Test Set...")

        # 1. Reverse One-Hot
        df = self._reverse_one_hot(df)

        # 2. Geometric
        df = self._add_geometric_features(df)

        return df


def process_data(df_train, df_test, load_cached_data=True):
    """
    Main entry point for feature engineering.
    Handles caching of the processed datasets to save time on subsequent runs.
    """
    # Define cache paths for the fully engineered features
    train_feats_path = os.path.join(config.WORKING_DIR, "train_feats.parquet")
    test_feats_path = os.path.join(config.WORKING_DIR, "test_feats.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(train_feats_path) and os.path.exists(test_feats_path):
            print(f"Loading engineered features from cache: {config.WORKING_DIR}")
            df_train_processed = pd.read_parquet(train_feats_path)
            df_test_processed = pd.read_parquet(test_feats_path)
            return df_train_processed, df_test_processed
        else:
            print("Engineered features cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing features from scratch...")

    # 2. Instantiate and run pipeline
    pipeline = FeaturePipeline()

    # Fit/Transform Train
    df_train_processed = pipeline.fit_transform(df_train)

    # Transform Test
    df_test_processed = pipeline.transform(df_test)

    # 3. Save to cache
    print(f"Saving engineered features to cache: {config.WORKING_DIR}")
    # Ensure directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    df_train_processed.to_parquet(train_feats_path, index=False)
    df_test_processed.to_parquet(test_feats_path, index=False)

    return df_train_processed, df_test_processed
