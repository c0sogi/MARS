import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from library import config


class FeaturePipeline:
    """
    A stateful feature engineering pipeline that transforms raw data into
    model-ready features using geometric calculations, PCA, and contextual grouping.
    """

    def __init__(self):
        self.pca = None
        self.scaler = None
        self.group_stats_map = {}

        # Base continuous columns to be used for PCA (before adding new ones)
        self.base_continuous_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]

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

    def _apply_pca(self, df, is_train=True):
        """
        Applies PCA to continuous features to capture rotational variance.
        """
        # Define columns to use for PCA: Base continuous + newly created Euclidean distance
        # We check existence to be safe
        candidates = self.base_continuous_cols + ["Euclidean_Distance_To_Hydrology"]
        cols_to_use = [c for c in candidates if c in df.columns]

        if not cols_to_use:
            return df

        X = df[cols_to_use].values

        # Standard Scaling (Required for PCA)
        if is_train:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # PCA Transformation
        if is_train:
            self.pca = PCA(
                n_components=config.PCA_N_COMPONENTS, random_state=config.SEED
            )
            X_pca = self.pca.fit_transform(X_scaled)
        else:
            X_pca = self.pca.transform(X_scaled)

        # Add PCA components as new features
        for i in range(config.PCA_N_COMPONENTS):
            df[f"PCA_{i+1}"] = X_pca[:, i].astype(np.float32)

        return df

    def _add_group_stats(self, df, is_train=True):
        """
        Computes contextual statistics (Mean/Std of Elevation) grouped by Soil_Type_Index.
        """
        group_col = "Soil_Type_Index"
        target_col = "Elevation"

        if group_col not in df.columns:
            return df

        if is_train:
            # Compute statistics on training data
            stats = df.groupby(group_col)[target_col].agg(["mean", "std"]).to_dict()
            self.group_stats_map["mean"] = stats["mean"]
            self.group_stats_map["std"] = stats["std"]

        # Apply mapping
        mean_map = self.group_stats_map.get("mean", {})
        std_map = self.group_stats_map.get("std", {})

        # Fallback values (Global stats)
        global_mean = df[target_col].mean()
        global_std = df[target_col].std()

        # Map and fill missing (e.g., if a soil type in test wasn't in train)
        df[f"{target_col}_Soil_Mean"] = (
            df[group_col].map(mean_map).fillna(global_mean).astype(np.float32)
        )
        df[f"{target_col}_Soil_Std"] = (
            df[group_col].map(std_map).fillna(global_std).astype(np.float32)
        )

        # Interaction feature: Deviation from expected value
        df[f"{target_col}_Soil_Deviation"] = (
            df[target_col] - df[f"{target_col}_Soil_Mean"]
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

        # 3. PCA (Fits on Train)
        df = self._apply_pca(df, is_train=True)

        # 4. Group Stats (Fits on Train)
        df = self._add_group_stats(df, is_train=True)

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

        # 3. PCA (Uses fitted model)
        df = self._apply_pca(df, is_train=False)

        # 4. Group Stats (Uses fitted stats)
        df = self._add_group_stats(df, is_train=False)

        return df


def process_data(df_train, df_val, df_test, load_cached_data=True):
    """
    Main entry point for feature engineering.
    Handles caching of the processed datasets to save time on subsequent runs.
    Updated to handle validation set separately.
    """
    # Define cache paths for the fully engineered features
    train_feats_path = os.path.join(config.WORKING_DIR, "train_feats.parquet")
    val_feats_path = os.path.join(config.WORKING_DIR, "val_feats.parquet")
    test_feats_path = os.path.join(config.WORKING_DIR, "test_feats.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_feats_path)
            and os.path.exists(val_feats_path)
            and os.path.exists(test_feats_path)
        ):
            print(f"Loading engineered features from cache: {config.WORKING_DIR}")
            df_train_processed = pd.read_parquet(train_feats_path)
            df_val_processed = pd.read_parquet(val_feats_path)
            df_test_processed = pd.read_parquet(test_feats_path)
            return df_train_processed, df_val_processed, df_test_processed
        else:
            print("Engineered features cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing features from scratch...")

    # 2. Instantiate and run pipeline
    pipeline = FeaturePipeline()

    # Fit/Transform Train
    df_train_processed = pipeline.fit_transform(df_train)

    # Transform Val (using fit from Train)
    print("Feature Engineering: Transforming Val Set...")
    df_val_processed = pipeline.transform(df_val)

    # Transform Test (using fit from Train)
    df_test_processed = pipeline.transform(df_test)

    # 3. Save to cache
    print(f"Saving engineered features to cache: {config.WORKING_DIR}")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    df_train_processed.to_parquet(train_feats_path, index=False)
    df_val_processed.to_parquet(val_feats_path, index=False)
    df_test_processed.to_parquet(test_feats_path, index=False)

    return df_train_processed, df_val_processed, df_test_processed
