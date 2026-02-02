import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
    RANDOM_SEED,
)
from library.preprocessing import DataPreprocessor, get_preprocessed_dataset
from library.data_manager import MaterialDataset

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DualXGBoostModel:
    """
    Wrapper class to manage training and inference of two XGBoost models,
    one for each target variable (formation_energy and bandgap_energy).
    """

    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        # Initialize a separate regressor for each target
        for target in TARGET_COLS:
            self.models[target] = xgb.XGBRegressor(**XGB_PARAMS)

    def train(self, load_cached_data=True, n_estimators=None):
        """
        Trains the models on the training set and evaluates on the validation set.

        Args:
            load_cached_data (bool): Whether to load features from cache.
            n_estimators (int, optional): Override the number of estimators (e.g. for debugging).
        """
        # Override n_estimators if provided
        if n_estimators is not None:
            for target in TARGET_COLS:
                self.models[target].set_params(n_estimators=n_estimators)

        # 1. Load and preprocess training data
        print("Loading training data...")
        # This will fit the preprocessor and save its state
        train_df = get_preprocessed_dataset(
            "train", self.preprocessor, load_cached_data=load_cached_data
        )

        # 2. Load and preprocess validation data
        print("Loading validation data...")
        # This will load the preprocessor state and transform val data
        val_df = get_preprocessed_dataset(
            "val", self.preprocessor, load_cached_data=load_cached_data
        )

        # 3. Load targets from metadata
        # The feature matrices (train_df, val_df) do not contain targets, so we fetch them from metadata.
        md = MaterialDataset()
        train_meta = md.load_metadata("train")
        val_meta = md.load_metadata("val")

        # 4. Align features and targets
        # Set 'id' as index for metadata to join with feature matrices
        y_train_full = train_meta.set_index("id")[TARGET_COLS]
        y_val_full = val_meta.set_index("id")[TARGET_COLS]

        # Inner join to ensure we only have targets for samples where features were successfully extracted
        train_combined = train_df.join(y_train_full, how="inner")
        val_combined = val_df.join(y_val_full, how="inner")

        X_train = train_combined.drop(columns=TARGET_COLS)
        y_train = train_combined[TARGET_COLS]

        X_val = val_combined.drop(columns=TARGET_COLS)
        y_val = val_combined[TARGET_COLS]

        metrics = {}
        print(
            f"Training on {len(X_train)} samples, validating on {len(X_val)} samples."
        )

        # 5. Train loop for each target
        for target in TARGET_COLS:
            print(f"\n--- Training model for Target: {target} ---")

            # Log transform the targets for training
            # z = log(1 + y)
            y_train_log = self.preprocessor.log_transform(y_train[target])
            y_val_log = self.preprocessor.log_transform(y_val[target])

            model = self.models[target]

            # Fit with early stopping
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_val, y_val_log)],
                verbose=100,  # Print progress every 100 rounds
            )

            # Evaluate
            preds_log = model.predict(X_val)

            # Calculate RMSLE
            # Since we trained on log1p(y), RMSE of predictions vs y_val_log is equivalent to RMSLE
            rmse = np.sqrt(mean_squared_error(y_val_log, preds_log))
            metrics[target] = rmse
            print(f"Validation RMSLE for {target}: {rmse:.10f}")

        # 6. Summary
        print("\n" + "=" * 30)
        print("Training Complete.")
        mean_rmsle = np.mean(list(metrics.values()))
        print(f"Overall Validation RMSLE (Mean): {mean_rmsle:.10f}")
        print("=" * 30 + "\n")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to load features from cache.
        """
        # 1. Load test data
        print("Loading test data...")
        test_df = get_preprocessed_dataset(
            "test", self.preprocessor, load_cached_data=load_cached_data
        )

        # 2. Prepare submission structure
        submission_df = pd.DataFrame(index=test_df.index)

        # 3. Prediction loop
        for target in TARGET_COLS:
            print(f"Predicting {target}...")
            model = self.models[target]

            # Predict (output is in log space)
            preds_log = model.predict(test_df)

            # Inverse transform to original space
            # y = exp(z) - 1
            preds = self.preprocessor.inverse_log_transform(preds_log)

            submission_df[target] = preds

        # 4. Format and Save
        # Move 'id' from index to column
        submission_df = submission_df.reset_index()

        # Ensure alignment with sample submission format
        if os.path.exists(SAMPLE_SUBMISSION_PATH):
            sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
            # Reorder columns to match sample
            cols = sample.columns.tolist()
            # Ensure we have all necessary columns
            if set(cols).issubset(submission_df.columns):
                submission_df = submission_df[cols]

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
