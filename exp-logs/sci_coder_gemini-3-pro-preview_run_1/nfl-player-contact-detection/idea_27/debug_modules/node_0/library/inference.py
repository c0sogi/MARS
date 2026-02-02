import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import CacheManager, seed_everything, compute_mcc
from library.data_factory import DataFactory
from library.feature_engineering import FeatureEngineer


class InferencePipeline:
    """
    Manages evaluation on validation set and inference on test set.
    Handles threshold optimization and submission generation.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        self.data_factory = DataFactory()
        self.feature_engineer = FeatureEngineer()
        seed_everything(Config.SEED)

    def _load_models(self):
        """
        Loads the trained Expert models from the cache.
        """
        models = {}
        model_names = ["lgbm", "xgb", "cat"]

        print("Loading Expert models for inference...")
        for name in model_names:
            filename = f"expert_{name}.joblib"
            if self.cache_manager.exists(filename):
                models[name] = self.cache_manager.load_joblib(filename)
            else:
                print(f"Warning: Model {filename} not found in cache. Skipping.")

        if not models:
            raise FileNotFoundError(
                "No expert models found in cache. Please run training first."
            )

        return models

    def optimize_threshold(self, load_cached_data=True):
        """
        Calculates the optimal decision threshold maximizing MCC on the validation set.
        Saves the best threshold to cache.

        Args:
            load_cached_data (bool): Whether to use cached features.

        Returns:
            float: The optimal threshold.
        """
        print("Starting threshold optimization on Validation set...")

        # 1. Load and Process Validation Data
        raw_val = self.data_factory.get_data("val", load_cached_data=load_cached_data)
        df_val = self.feature_engineer.process_features(
            raw_val, "val", load_cached_data=load_cached_data
        )

        X_val = df_val[Config.FEATURE_COLS]
        y_val = df_val["contact"].values

        # 2. Load Models
        models = self._load_models()

        # 3. Generate Ensemble Probabilities
        avg_probs = np.zeros(len(X_val))
        for name, model in models.items():
            probs = model.predict_proba(X_val)[:, 1]
            avg_probs += probs

        avg_probs /= len(models)

        # 4. Grid Search for Best Threshold
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range: 0.1 to 0.9
        thresholds = np.arange(0.1, 0.91, 0.01)

        for thresh in thresholds:
            preds = (avg_probs >= thresh).astype(int)
            mcc = compute_mcc(y_val, preds)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Optimization Complete. Best Threshold: {best_thresh}")
        print(f"Best Validation MCC: {best_mcc}")

        # 5. Save Threshold
        self.cache_manager.save_npy(np.array([best_thresh]), "best_threshold.npy")

        return best_thresh

    def predict_test_set(self, load_cached_data=True, use_optimized_threshold=True):
        """
        Generates predictions for the test set using the ensemble of models.
        Saves the result to submission.csv.

        Args:
            load_cached_data (bool): Whether to use cached features.
            use_optimized_threshold (bool): If True, loads best_threshold.npy.
                                            Otherwise defaults to 0.5.
        """
        print("Starting inference on Test set...")

        # 1. Load and Process Test Data
        raw_test = self.data_factory.get_data("test", load_cached_data=load_cached_data)
        df_test = self.feature_engineer.process_features(
            raw_test, "test", load_cached_data=load_cached_data
        )

        X_test = df_test[Config.FEATURE_COLS]

        # 2. Load Models
        models = self._load_models()

        # 3. Generate Ensemble Probabilities
        print(f"Predicting with {len(models)} models...")
        avg_probs = np.zeros(len(X_test))
        for name, model in models.items():
            probs = model.predict_proba(X_test)[:, 1]
            avg_probs += probs

        avg_probs /= len(models)

        # 4. Determine Threshold
        threshold = 0.5
        if use_optimized_threshold:
            if self.cache_manager.exists("best_threshold.npy"):
                threshold = self.cache_manager.load_npy("best_threshold.npy")[0]
                print(f"Using optimized threshold: {threshold}")
            else:
                print("Optimized threshold not found. Using default: 0.5")
        else:
            print("Using default threshold: 0.5")

        # 5. Apply Threshold
        predictions = (avg_probs >= threshold).astype(int)

        # 6. Generate Submission File
        # We need to ensure the submission matches the sample_submission.csv format exactly.
        # df_test contains 'contact_id' and our predictions.

        submission_df = df_test[["contact_id"]].copy()
        submission_df["contact"] = predictions

        # Load template to ensure all IDs are present and in order
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)

            # Merge predictions into the template
            # We use left merge on the template to preserve its rows/order
            final_submission = sample_sub[["contact_id"]].merge(
                submission_df, on="contact_id", how="left"
            )

            # Fill missing predictions with 0 (no contact)
            # This handles cases where feature engineering might have filtered rows (unlikely for test)
            # or data was missing.
            final_submission["contact"] = (
                final_submission["contact"].fillna(0).astype(int)
            )
        else:
            # Fallback if sample_submission not found (e.g. custom run)
            print(
                "Warning: sample_submission.csv not found. Using generated predictions directly."
            )
            final_submission = submission_df

        # 7. Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total Predictions: {len(final_submission)}")
        print(f"Positive Predictions: {final_submission['contact'].sum()}")
