import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SUBMISSION_DIR,
    SEED,
    SAMPLE_SUBMISSION_PATH,
)
from library.feature_engineering import FeatureExtractor
from library.model import DirectionalLGBM
from library.utils import angular_dist_score, cartesian_to_spherical


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with feature extractor and model wrapper.
        """
        self.extractor = FeatureExtractor()
        self.model = DirectionalLGBM()
        self.model_path = WORKING_DIR / "model.pkl"

        # Ensure working directory exists
        WORKING_DIR.mkdir(parents=True, exist_ok=True)
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, meta_path, mode, sample_size=None, load_cached_data=True):
        """
        Loads metadata and extracts features.

        Args:
            meta_path (Path): Path to the metadata parquet file.
            mode (str): 'train' or 'test'.
            sample_size (int, optional): Number of events to sample.
            load_cached_data (bool): Whether to use cached features.

        Returns:
            tuple: (X, y, ids)
        """
        print(f"Loading metadata from {meta_path}...")
        df_meta = pd.read_parquet(meta_path)

        if sample_size is not None and sample_size < len(df_meta):
            print(f"Sampling {sample_size} events from {len(df_meta)} total events.")
            df_meta = df_meta.sample(n=sample_size, random_state=SEED).reset_index(
                drop=True
            )

        print(f"Extracting features for {len(df_meta)} events (Mode: {mode})...")
        X, y, ids = self.extractor.extract_features(
            df_meta, mode=mode, load_cached_data=load_cached_data
        )

        return X, y, ids

    def train_and_evaluate(self, sample_size=None, load_cached_data=True):
        """
        Runs the training pipeline: Data loading, Training, and Evaluation.

        Args:
            sample_size (int, optional): Limit training data size for debugging.
            load_cached_data (bool): Use cached features if available.
        """
        print("Starting Training Pipeline...")

        # 1. Load Data
        X_train, y_train, _ = self.load_dataset(
            TRAIN_META_PATH,
            mode="train",
            sample_size=sample_size,
            load_cached_data=load_cached_data,
        )

        # For validation, we usually want the full set unless debugging
        val_sample_size = sample_size if sample_size is not None else None
        X_val, y_val, _ = self.load_dataset(
            VAL_META_PATH,
            mode="train",  # Validation data is technically in the 'train' folder structure
            sample_size=val_sample_size,
            load_cached_data=load_cached_data,
        )

        if len(X_train) == 0:
            raise ValueError("Training data is empty.")

        # 2. Train Model
        print("Training model...")
        self.model.fit(X_train, y_train, X_val, y_val)

        # Save model
        self.model.save(self.model_path)

        # 3. Evaluate
        print("Evaluating on validation set...")
        pred_azimuth, pred_zenith = self.model.predict(X_val)

        # Reconstruct ground truth spherical coordinates from Cartesian targets
        # y_val columns are [target_x, target_y, target_z]
        true_azimuth, true_zenith = cartesian_to_spherical(
            y_val[:, 0], y_val[:, 1], y_val[:, 2]
        )

        score = angular_dist_score(true_azimuth, true_zenith, pred_azimuth, pred_zenith)

        print(f"Validation Mean Angular Error: {score}")

        # Clean up memory
        del X_train, y_train, X_val, y_val
        gc.collect()

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Use cached features if available.
        """
        print("Starting Submission Generation...")

        # 1. Load Model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Model file not found. Run train_and_evaluate first."
            )

        self.model.load(self.model_path)

        # 2. Load Test Data
        # We process the test set. Note: Test metadata might be large,
        # but FeatureExtractor handles batching logic internally for memory efficiency during extraction.
        # However, it returns the full concatenated array.
        # With 13M events, X will be ~1.2GB, which fits in RAM.
        X_test, _, ids_test = self.load_dataset(
            TEST_META_PATH,
            mode="test",
            sample_size=None,  # Always predict on full test set
            load_cached_data=load_cached_data,
        )

        if len(X_test) == 0:
            raise ValueError("Test data is empty.")

        # 3. Predict
        print("Predicting on test set...")
        pred_azimuth, pred_zenith = self.model.predict(X_test)

        # 4. Create Submission DataFrame
        print("Creating submission file...")
        submission_df = pd.DataFrame(
            {"event_id": ids_test, "azimuth": pred_azimuth, "zenith": pred_zenith}
        )

        # Sort by event_id to match sample submission structure (good practice)
        submission_df = submission_df.sort_values("event_id")

        # Save to CSV
        out_path = SUBMISSION_DIR / "submission.csv"
        submission_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")

        # Clean up
        del X_test, ids_test, submission_df
        gc.collect()
