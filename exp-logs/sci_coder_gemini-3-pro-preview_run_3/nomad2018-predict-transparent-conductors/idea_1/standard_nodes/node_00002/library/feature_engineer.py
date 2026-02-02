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
        self.poly_cols = ["percent_atom_al", "percent_atom_ga", "percent_atom_in"]
        self.cat_cols = Config.CAT_COLS

        # Numerical columns are everything in Config.NUM_COLS not in poly_cols, plus geometry cols
        self.num_cols = [
            c for c in Config.NUM_COLS if c not in self.poly_cols
        ] + Config.GEO_COLS

        # Initialize transformers
        # Handle unknown categories gracefully, though spacegroups are likely fixed
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Degree 2 polynomials for composition (interaction terms are important for bowing parameters)
        self.poly = PolynomialFeatures(
            degree=Config.POLYNOMIAL_DEGREE, include_bias=False
        )

        # Imputer for numerical columns (e.g., if geometry extraction fails for some rows)
        self.imputer = SimpleImputer(strategy="mean")

        # Scaler for the final feature vector
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

        # 2. Fit Polynomial Features
        self.poly.fit(df[self.poly_cols])

        # 3. Fit Imputer
        # Ensure we handle cases where columns might be missing by reindexing if necessary,
        # though Config guarantees existence in metadata.
        self.imputer.fit(df[self.num_cols])

        # 4. Generate intermediate data to fit the Scaler
        X_cat = self.ohe.transform(df[self.cat_cols])
        X_poly = self.poly.transform(df[self.poly_cols])
        X_num = self.imputer.transform(df[self.num_cols])

        X_combined = np.hstack([X_cat, X_poly, X_num])

        # 5. Fit Scaler
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
        # Get feature names for OHE
        cat_names = self.ohe.get_feature_names_out(self.cat_cols)

        # 2. Transform Polynomial
        X_poly = self.poly.transform(df[self.poly_cols])
        poly_names = self.poly.get_feature_names_out(self.poly_cols)

        # 3. Transform Numerical (Impute)
        X_num = self.imputer.transform(df[self.num_cols])
        num_names = self.num_cols

        # 4. Concatenate
        X_combined = np.hstack([X_cat, X_poly, X_num])

        # 5. Scale
        X_scaled = self.scaler.transform(X_combined)

        # 6. Construct DataFrame
        all_feature_names = list(cat_names) + list(poly_names) + list(num_names)
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
