import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.data_manager import DataManager
from library.models import LGBMWrapper, XGBWrapper, Ensemble
from library.utils import setup_logging


class InferenceManager:
    """
    Manages the inference pipeline for the Relative-Velocity-Aligned Anchored-Mining Ensemble.
    Handles loading models, applying gating logic, and generating the final submission.
    """

    def __init__(self, config=Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")

    def load_expert_models(self):
        """
        Loads the trained expert models from the working directory.

        Returns:
            list: A list of loaded model wrappers (LGBMWrapper, XGBWrapper).
        """
        models = []

        # Load LightGBM Expert
        lgbm_path = os.path.join(self.models_dir, "expert_lgbm.joblib")
        if os.path.exists(lgbm_path):
            print(f"Loading LightGBM Expert from {lgbm_path}...")
            lgbm_model = LGBMWrapper().load(lgbm_path)
            models.append(lgbm_model)
        else:
            print(f"Warning: LightGBM model not found at {lgbm_path}")

        # Load XGBoost Expert
        xgb_path = os.path.join(self.models_dir, "expert_xgb.joblib")
        if os.path.exists(xgb_path):
            print(f"Loading XGBoost Expert from {xgb_path}...")
            xgb_model = XGBWrapper().load(xgb_path)
            models.append(xgb_model)
        else:
            print(f"Warning: XGBoost model not found at {xgb_path}")

        if not models:
            raise FileNotFoundError(
                "No trained expert models found in models directory."
            )

        return models

    def load_threshold(self):
        """
        Loads the optimized decision threshold.

        Returns:
            float: The threshold value. Defaults to 0.5 if file not found.
        """
        thresh_path = os.path.join(self.models_dir, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = np.load(thresh_path)[0]
            print(f"Loaded optimized threshold: {threshold}")
            return threshold
        else:
            print(
                f"Warning: Threshold file not found at {thresh_path}. Defaulting to 0.5."
            )
            return 0.5

    def predict_test_set(self, load_cached_data=True):
        """
        Executes the full inference pipeline:
        1. Loads test data (features).
        2. Loads models and threshold.
        3. Predicts probabilities using the Ensemble.
        4. Applies Quadratic Gating (forces 0 probability for gated rows).
        5. Applies Threshold.
        6. Saves submission file.

        Args:
            load_cached_data (bool): Whether to use cached feature files.
        """
        setup_logging()
        print("\n--- Starting Inference Pipeline ---")

        # 1. Load Test Data
        # FeatureEngineer.process_test returns all rows with a 'gating_active' column
        df_test = self.data_manager.get_test_data(load_cached_data=load_cached_data)

        # Extract Gating Mask (1 = keep, 0 = force no contact)
        if "gating_active" in df_test.columns:
            gating_mask = df_test["gating_active"].values
        else:
            print(
                "Warning: 'gating_active' column not found. Assuming all rows are active."
            )
            gating_mask = np.ones(len(df_test))

        # Prepare Features (X)
        # get_X_y strips metadata columns including gating_active
        X_test, _ = self.data_manager.get_X_y(df_test)

        print(f"Test Data Shape: {X_test.shape}")

        # 2. Load Resources
        models = self.load_expert_models()
        threshold = self.load_threshold()

        # 3. Ensemble Prediction
        ensemble = Ensemble(models)
        print("Generating raw probabilities...")
        raw_probs = ensemble.predict(X_test)

        # 4. Apply Gating
        # If gating_active is 0, the pair is physically too far/divergent, so prob -> 0
        print("Applying Relaxed Quadratic Gating...")
        gated_probs = raw_probs * gating_mask

        # 5. Apply Threshold
        print(f"Applying decision threshold: {threshold}")
        predictions = (gated_probs >= threshold).astype(int)

        # 6. Generate Submission
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Ensure output directory exists
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
        print(f"Total Predictions: {len(submission)}")
        print(
            f"Positive Predictions: {submission['contact'].sum()} ({submission['contact'].mean():.4%})"
        )

        return submission
