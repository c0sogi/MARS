import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from library.config import Config


class FeaturePipeline:
    def __init__(self):
        """
        Initializes the feature engineering pipeline with specific transformers
        for different feature groups.
        """
        # Define feature groups
        self.cat_cols = Config.CAT_COLS
        # Treat all numerical columns together, no polynomial expansion needed for XGBoost
        self.num_cols = Config.NUM_COLS + Config.GEO_COLS

        # Initialize transformers
        # Handle unknown categories gracefully
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Imputer for numerical columns
        self.imputer = SimpleImputer(strategy="mean")

        # Scaler for the final feature vector (optional for trees but good practice)
        self.scaler = StandardScaler()

        self.feature_names = None

    def fit(self, df: pd.DataFrame):
        """
        Fits the transformers to the training data.

        Args:
            df (pd.DataFrame): Training data containing raw features.
        """
        # 1. Fit One-Hot Encoder
        self.ohe.fit(df[self.cat_cols])

        # 2. Fit Imputer
        self.imputer.fit(df[self.num_cols])

        # 3. Generate intermediate data to fit the Scaler
        X_cat = self.ohe.transform(df[self.cat_cols])
        X_num = self.imputer.transform(df[self.num_cols])

        X_combined = np.hstack([X_cat, X_num])

        # 4. Fit Scaler
        self.scaler.fit(X_combined)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the fitted transformers.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Processed features with ID and Targets (if present).
        """
        # 1. Transform Categorical
        X_cat = self.ohe.transform(df[self.cat_cols])
        cat_names = self.ohe.get_feature_names_out(self.cat_cols)

        # 2. Transform Numerical (Impute)
        X_num = self.imputer.transform(df[self.num_cols])
        num_names = self.num_cols

        # 3. Concatenate
        X_combined = np.hstack([X_cat, X_num])

        # 4. Scale
        X_scaled = self.scaler.transform(X_combined)

        # 5. Construct DataFrame
        all_feature_names = list(cat_names) + list(num_names)
        X_df = pd.DataFrame(X_scaled, columns=all_feature_names, index=df.index)

        # 7. Re-attach ID and Targets
        if Config.ID_COL in df.columns:
            X_df.insert(0, Config.ID_COL, df[Config.ID_COL])

        for target in Config.TARGET_COLS:
            if target in df.columns:
                X_df[target] = df[target]

        return X_df

    def process_and_cache(
        self,
        df: pd.DataFrame,
        cache_path: str,
        is_training: bool = True,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Orchestrates the fitting (if training), transformation, and caching of data.

        Args:
            df (pd.DataFrame): Input dataframe (raw metadata + geometry features).
            cache_path (str): Path to save/load the processed parquet file.
            is_training (bool): If True, fits the pipeline on this data.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The processed feature dataframe.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Logic Flow 1: Try to load
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading processed features from cache: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Logic Flow 2: Compute
        print(f"Processing features (Training={is_training})...")

        if is_training:
            self.fit(df)

        # Transform
        processed_df = self.transform(df)

        # Logic Flow 3: Save
        try:
            processed_df.to_parquet(cache_path, index=False)
            print(f"Saved processed features to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return processed_df
