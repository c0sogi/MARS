import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    RADIOMICS_FEATURES,
    TABULAR_NUMERICAL_FEATURES,
    TABULAR_CATEGORICAL_FEATURES,
    TARGET_COL,
    CONFIDENCE_COL,
    RANDOM_STATE,
)
from library.image_processing import process_all_scans


class FeatureEngineer:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define file paths for cached numpy arrays
        self.cache_files = {
            "X_train": os.path.join(self.cache_dir, "X_train.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.npy"),
            "test_meta": os.path.join(self.cache_dir, "test_meta.parquet"),
        }

        # Initialize Preprocessing Pipeline
        # Numerical: Impute mean (safety) -> Scale
        # Categorical: Impute constant -> OneHot

        # Combine defined numerical features with radiomics features for scaling
        self.numeric_features = TABULAR_NUMERICAL_FEATURES + RADIOMICS_FEATURES
        self.categorical_features = TABULAR_CATEGORICAL_FEATURES

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="missing"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    self.categorical_features,
                ),
            ],
            verbose_feature_names_out=False,
        )

    def _derive_baseline_features(self, df):
        """
        For train/val sets, we need to create 'Baseline_FVC', 'Baseline_Percent',
        and 'Baseline_Weeks' columns. We assume the baseline is the earliest
        recorded visit (min Weeks) for each patient.
        """
        # Sort by Patient and Weeks to ensure we get the earliest visit first
        df_sorted = df.sort_values(by=["Patient", "Weeks"])

        # Group by Patient and take the first record
        baseline_df = df_sorted.groupby("Patient").first().reset_index()

        # Select relevant columns and rename
        baseline_df = baseline_df[["Patient", "FVC", "Percent", "Weeks"]]
        baseline_df = baseline_df.rename(
            columns={
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Weeks": "Baseline_Weeks",
            }
        )

        # Merge back to the original dataframe
        df_merged = pd.merge(df, baseline_df, on="Patient", how="left")
        return df_merged

    def _prepare_single_df(self, df, radiomics_df, is_test=False):
        """
        Merges clinical data with radiomics and ensures baseline columns exist.
        """
        # 1. Add Baseline Features if not test set
        # (Test set already has these columns from metadata generation)
        if not is_test:
            df = self._derive_baseline_features(df)

        # 2. Merge Radiomics Features
        # Left merge to keep all clinical records, filling missing radiomics if any
        df = pd.merge(df, radiomics_df, on="Patient", how="left")

        # Fill missing radiomics with 0 or mean (handled by imputer later, but 0 is safe for now)
        for col in RADIOMICS_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def load_datasets(self, load_cached_data=True):
        """
        Main entry point. Loads data, processes it, and returns X, y arrays.
        Implements caching using .npy and .parquet files.
        """
        # 1. Check if cache exists
        all_cached = all(os.path.exists(p) for p in self.cache_files.values())

        if load_cached_data and all_cached:
            print(f"Loading processed datasets from cache: {self.cache_dir}")
            X_train = np.load(self.cache_files["X_train"])
            y_train = np.load(self.cache_files["y_train"])
            X_val = np.load(self.cache_files["X_val"])
            y_val = np.load(self.cache_files["y_val"])
            X_test = np.load(self.cache_files["X_test"])
            test_meta = pd.read_parquet(self.cache_files["test_meta"])
            return X_train, y_train, X_val, y_val, X_test, test_meta

        # 2. If not cached, process from scratch
        print("Processing datasets from scratch...")

        # Load Metadata
        train_df = pd.read_csv(TRAIN_METADATA_PATH)
        val_df = pd.read_csv(VAL_METADATA_PATH)
        test_df = pd.read_csv(TEST_METADATA_PATH)

        # Load/Extract Radiomics (this function handles its own caching)
        # We combine all patients to process efficiently
        all_meta = pd.concat([train_df, val_df, test_df], ignore_index=True)
        radiomics_df = process_all_scans(all_meta, load_cached_data=load_cached_data)

        # Process each split
        train_df = self._prepare_single_df(train_df, radiomics_df, is_test=False)
        val_df = self._prepare_single_df(val_df, radiomics_df, is_test=False)
        test_df = self._prepare_single_df(test_df, radiomics_df, is_test=True)

        # Fit Preprocessor on Training Data ONLY
        print("Fitting feature preprocessor on training data...")
        self.preprocessor.fit(train_df)

        # Transform all splits
        X_train = self.preprocessor.transform(train_df)
        X_val = self.preprocessor.transform(val_df)
        X_test = self.preprocessor.transform(test_df)

        # Extract Targets
        y_train = train_df[TARGET_COL].values.astype(np.float32)
        y_val = val_df[TARGET_COL].values.astype(np.float32)
        # Test target is placeholder, so we don't return y_test usually,
        # but we return test_meta which contains Patient_Week for submission

        # Save to Cache
        print(f"Saving processed datasets to cache: {self.cache_dir}")
        np.save(self.cache_files["X_train"], X_train)
        np.save(self.cache_files["y_train"], y_train)
        np.save(self.cache_files["X_val"], X_val)
        np.save(self.cache_files["y_val"], y_val)
        np.save(self.cache_files["X_test"], X_test)
        test_df.to_parquet(self.cache_files["test_meta"], index=False)

        return X_train, y_train, X_val, y_val, X_test, test_df
