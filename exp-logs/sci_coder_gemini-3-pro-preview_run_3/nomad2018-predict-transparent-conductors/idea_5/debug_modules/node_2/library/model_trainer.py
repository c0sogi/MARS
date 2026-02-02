import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    SUBMISSION_FILE_PATH,
    WORKING_DIR,
    RANDOM_SEED,
)
from library.data_loader import load_metadata, load_geometries
from library.physical_descriptors import PhysicalFeaturizer
from library.gnn_processor import MatGLEmbedder


class EnergyPredictor:
    """
    A class to handle the end-to-end workflow of training XGBoost models
    on hybrid features (Physical + Chemically-Resolved MatGL Embeddings)
    and generating predictions.
    """

    def __init__(self):
        self.physical_featurizer = PhysicalFeaturizer()
        self.matgl_embedder = MatGLEmbedder()
        self.models = {}  # Dictionary to store trained models for each target

    def _get_combined_features(self, split, max_samples=None, load_cached_data=True):
        """
        Internal method to load metadata, generate all features, and combine them.
        Implements caching for the combined dataframe.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Cache path for the combined dataframe
        # We include max_samples in filename if it's set, to avoid loading partial data as full
        suffix = "" if max_samples is None else f"_sample_{max_samples}"
        cache_path = os.path.join(
            WORKING_DIR, f"{split}_combined_features{suffix}.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading combined features for {split} from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing features.")

        # 1. Load Metadata
        df_meta = load_metadata(
            split=split, max_samples=max_samples, load_cached_data=load_cached_data
        )

        # 2. Load Geometries
        atoms_list = load_geometries(df_meta)

        # 3. Generate Physical Features
        # Note: PhysicalFeaturizer handles its own caching logic, but we pass load_cached_data
        df_phys = self.physical_featurizer.featurize(
            df_meta, atoms_list, split, load_cached_data=load_cached_data
        )

        # 4. Generate MatGL Embeddings
        # Note: MatGLEmbedder handles its own caching logic
        df_matgl = self.matgl_embedder.generate_chemically_resolved_embeddings(
            atoms_list, split, load_cached_data=load_cached_data
        )

        # 5. Combine Features
        # We concatenate column-wise. We must ensure indices align (reset_index was done in load_metadata)
        # We drop duplicate columns that might exist in both df_meta and df_phys (like composition)
        # to keep the dataframe clean, although XGBoost handles duplicates fine.

        # Identify columns to drop from df_phys that are already in df_meta
        duplicate_cols = df_phys.columns.intersection(df_meta.columns)
        df_phys_clean = df_phys.drop(columns=duplicate_cols)

        # Concatenate
        df_combined = pd.concat([df_meta, df_phys_clean, df_matgl], axis=1)

        # 6. Save to Cache
        try:
            df_combined.to_parquet(cache_path, index=False)
            print(f"Saved combined features for {split} to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save combined features cache: {e}")

        return df_combined

    def prepare_data(self, split, max_samples=None, load_cached_data=True):
        """
        Prepares X (features) and y (targets) for a given split.

        Args:
            split (str): 'train', 'val', or 'test'.
            max_samples (int): Limit number of samples.
            load_cached_data (bool): Whether to use caching.

        Returns:
            tuple: (X, y, ids)
        """
        df = self._get_combined_features(split, max_samples, load_cached_data)

        # Define feature columns:
        # Exclude ID, file_path, and targets from the feature set X
        exclude_cols = ["id", "file_path"] + TARGET_COLS
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]

        # Extract targets if they exist (train/val), else None
        y = None
        if all(col in df.columns for col in TARGET_COLS):
            y = df[TARGET_COLS]

        ids = df["id"]

        print(f"Prepared data for {split}: X shape={X.shape}")
        return X, y, ids

    def train_model(self, max_samples=None, load_cached_data=True):
        """
        Trains XGBoost models for formation energy and bandgap energy.
        Applies log(1+y) transformation to targets.
        """
        print("\n" + "=" * 40)
        print(" TRAINING MODELS ")
        print("=" * 40)

        # 1. Prepare Data
        print("\n--- Preparing Training Data ---")
        X_train, y_train, _ = self.prepare_data("train", max_samples, load_cached_data)

        print("\n--- Preparing Validation Data ---")
        X_val, y_val, _ = self.prepare_data("val", max_samples, load_cached_data)

        # 2. Train for each target
        for target in TARGET_COLS:
            print(f"\nTraining XGBoost for target: {target}")

            # Log-transform targets to align with RMSLE metric
            # z = log(1 + y)
            y_train_log = np.log1p(y_train[target])
            y_val_log = np.log1p(y_val[target])

            # Initialize model
            model = xgb.XGBRegressor(**XGB_PARAMS)

            # Fit model with early stopping
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
                verbose=500,  # Print every 500 rounds
            )

            # Evaluate
            val_preds_log = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val_log, val_preds_log))

            # Since we predict log(1+y), RMSE on log scale is effectively RMSLE on original scale
            print(f"Validation RMSLE for {target}: {val_rmse}")

            # Store model
            self.models[target] = model

    def predict_and_submit(self, max_samples=None, load_cached_data=True):
        """
        Generates predictions for the test set, inverse transforms them,
        and saves the submission file.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")

        print("\n" + "=" * 40)
        print(" GENERATING SUBMISSION ")
        print("=" * 40)

        print("\n--- Preparing Test Data ---")
        X_test, _, ids = self.prepare_data("test", max_samples, load_cached_data)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": ids})

        for target in TARGET_COLS:
            print(f"Predicting {target}...")
            model = self.models[target]

            # Predict in log scale
            preds_log = model.predict(X_test)

            # Inverse transform: y = exp(z) - 1
            preds = np.expm1(preds_log)

            # Sanity check: Ensure no negative values (physics constraint)
            preds = np.maximum(preds, 0)

            submission_df[target] = preds

        # Save submission
        submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"\nSubmission saved to {SUBMISSION_FILE_PATH}")
        print("First 5 rows of submission:")
        print(submission_df.head())
