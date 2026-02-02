import pandas as pd
import numpy as np
import os
from library.config import PathConfig, ModelConfig
from library.data_loader import DataLoader
from library.model_factory import LGBMWrapper, XGBWrapper


class InferencePipeline:
    """
    Manages the inference workflow for the Dual-Scout Physics-Enhanced Mining Ensemble.
    Loads processed test features, trained expert models, and the optimized threshold
    to generate the final submission file.
    """

    def __init__(self):
        self.paths = PathConfig()
        self.model_config = ModelConfig()
        self.loader = DataLoader()

    def predict_test_set(self, load_cached=True):
        """
        Executes the full inference pipeline:
        1. Loads/Generates test features.
        2. Loads trained Expert LightGBM and XGBoost models.
        3. Loads the optimized decision threshold.
        4. Computes ensemble probabilities.
        5. Applies threshold and saves submission.csv.

        Args:
            load_cached (bool): If True, attempts to load features from cache.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        print("\n--- Starting Inference Pipeline ---")

        # 1. Load Test Data
        # The DataLoader delegates to FeatureEngineer, which handles caching of the parquet file
        print("Preparing test dataset...")
        test_df = self.loader.prepare_test_dataset(load_cached=load_cached)

        # Extract feature matrix X (target y is None for test set)
        X_test, _ = self.loader.split_features_target(test_df)

        # 2. Load Trained Expert Models
        print("Loading Expert Models...")

        # Initialize wrappers with config parameters (required for instantiation)
        # Then load the trained artifacts from disk
        expert_lgbm = LGBMWrapper(self.model_config.EXPERT_LGBM_PARAMS)
        try:
            expert_lgbm.load(self.paths.EXPERT_LGBM_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Expert LightGBM model not found at {self.paths.EXPERT_LGBM_PATH}. Please run training first."
            )

        expert_xgb = XGBWrapper(self.model_config.EXPERT_XGB_PARAMS)
        try:
            expert_xgb.load(self.paths.EXPERT_XGB_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Expert XGBoost model not found at {self.paths.EXPERT_XGB_PATH}. Please run training first."
            )

        # 3. Load Optimized Threshold
        threshold_path = os.path.join(self.paths.WORKING_DIR, "best_threshold.npy")
        if os.path.exists(threshold_path):
            best_threshold = np.load(threshold_path)[0]
            print(f"Loaded optimized threshold: {best_threshold}")
        else:
            print("Warning: Threshold file not found. Defaulting to 0.5.")
            best_threshold = 0.5

        # 4. Generate Predictions (Ensemble)
        print("Generating predictions...")

        # Get probabilities from Expert A (LightGBM)
        probs_lgbm = expert_lgbm.predict(X_test)

        # Get probabilities from Expert B (XGBoost)
        probs_xgb = expert_xgb.predict(X_test)

        # Compute Unweighted Average
        probs_ensemble = (probs_lgbm + probs_xgb) / 2.0

        # Apply Threshold
        binary_predictions = (probs_ensemble >= best_threshold).astype(int)

        # 5. Format and Save Submission
        print("Formatting submission...")
        submission = pd.DataFrame(
            {"contact_id": test_df["contact_id"], "contact": binary_predictions}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.paths.SUBMISSION_FILE_PATH), exist_ok=True)

        print(f"Saving submission to {self.paths.SUBMISSION_FILE_PATH}...")
        submission.to_csv(self.paths.SUBMISSION_FILE_PATH, index=False)
        print("Submission saved successfully.")

        return submission
