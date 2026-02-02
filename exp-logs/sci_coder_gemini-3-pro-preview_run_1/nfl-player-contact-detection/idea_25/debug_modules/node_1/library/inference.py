import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.features import FeatureEngineer
from library.models import TriEnsemble


class InferenceManager:
    """
    Manages the inference process including model loading, threshold optimization,
    and submission generation.
    """

    def __init__(self):
        """
        Initialize the InferenceManager.
        Instantiates the FeatureEngineer and loads the trained TriEnsemble models.
        """
        self.feature_engineer = FeatureEngineer()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Load the trained expert ensemble
        print("Initializing InferenceManager and loading models...")
        self.ensemble = TriEnsemble()
        try:
            self.ensemble.load_models(prefix="expert")
        except FileNotFoundError:
            print("Warning: Expert models not found. Ensure training has completed.")

    def optimize_threshold(self, load_cached_data=True):
        """
        Optimizes the decision threshold by maximizing MCC on the validation set.

        Args:
            load_cached_data (bool): Whether to use cached validation features.

        Returns:
            float: The optimal threshold.
        """
        print("Starting threshold optimization on validation set...")

        # 1. Load Validation Data
        # FeatureEngineer handles caching internally
        df_val = self.feature_engineer.process_data(
            dataset_type="val", load_cached_data=load_cached_data
        )

        # 2. Apply Gating Logic
        # We only predict on pairs that pass the quadratic reachability filter.
        # Others are automatically assumed to be 0 (No Contact).
        gating_mask = df_val["quadratic_min_dist"] < Config.GATING_THRESHOLD

        # Select features for the gated subset
        X_val_gated = df_val.loc[gating_mask, Config.FEATURES]

        # 3. Generate Probabilities
        print(f"Predicting on {len(X_val_gated)} gated validation samples...")
        if len(X_val_gated) > 0:
            probs_gated = self.ensemble.predict_proba(X_val_gated)[:, 1]
        else:
            probs_gated = np.array([])

        # Reconstruct full probability vector
        # Initialize with 0.0 (default for non-gated)
        y_probs_full = np.zeros(len(df_val))
        y_probs_full[gating_mask] = probs_gated

        y_true_full = df_val["contact"].values

        # 4. Grid Search for Best Threshold
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_mcc = -1.0
        best_th = 0.5

        for th in thresholds:
            y_pred = (y_probs_full >= th).astype(int)
            mcc = matthews_corrcoef(y_true_full, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        print(
            f"Optimization Complete. Best Threshold: {best_th:.16f}, Best MCC: {best_mcc:.16f}"
        )

        # 5. Save Threshold
        threshold_path = os.path.join(Config.CACHE_MODELS, "best_threshold.npy")
        np.save(threshold_path, np.array([best_th]))
        print(f"Best threshold saved to {threshold_path}")

        return best_th

    def generate_predictions(self, threshold=None, load_cached_data=True):
        """
        Generates predictions for the test set and creates the submission file.

        Args:
            threshold (float, optional): Decision threshold. If None, loads from cache.
            load_cached_data (bool): Whether to use cached test features.
        """
        print("Starting submission generation...")

        # 1. Determine Threshold
        if threshold is None:
            threshold_path = os.path.join(Config.CACHE_MODELS, "best_threshold.npy")
            if os.path.exists(threshold_path):
                threshold = float(np.load(threshold_path)[0])
                print(f"Loaded best threshold from cache: {threshold:.16f}")
            else:
                threshold = 0.5
                print(f"Threshold cache not found. Using default: {threshold}")
        else:
            print(f"Using provided threshold: {threshold:.16f}")

        # 2. Load Test Data
        df_test = self.feature_engineer.process_data(
            dataset_type="test", load_cached_data=load_cached_data
        )

        # 3. Apply Gating Logic
        gating_mask = df_test["quadratic_min_dist"] < Config.GATING_THRESHOLD

        X_test_gated = df_test.loc[gating_mask, Config.FEATURES]

        # 4. Predict
        print(f"Predicting on {len(X_test_gated)} gated test samples...")
        if len(X_test_gated) > 0:
            probs_gated = self.ensemble.predict_proba(X_test_gated)[:, 1]
        else:
            probs_gated = np.array([])

        # 5. Reconstruct Full Predictions
        # Initialize with 0.0
        y_probs_full = np.zeros(len(df_test))
        y_probs_full[gating_mask] = probs_gated

        # Apply Threshold
        predictions = (y_probs_full >= threshold).astype(int)

        # 6. Create Submission File
        self.create_submission(df_test, predictions)

    def create_submission(self, df_test, predictions):
        """
        Formats and saves the submission CSV.

        Args:
            df_test (pd.DataFrame): Test metadata containing contact_id.
            predictions (np.ndarray): Binary predictions.
        """
        print("Formatting submission file...")

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        submission_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH}")
        print(f"Total rows: {len(submission_df)}")
        print(f"Positive predictions: {submission_df['contact'].sum()}")
