import os
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, SUBMISSION_PATH, SEED
from library.utils import seed_everything
from library.data_loader import DataLoader
from library.model_factory import EnsemblePredictor
from library.training_pipeline import NON_FEATURE_COLS


class InferencePipeline:
    """
    Manages the inference process for the Kinematically-Aligned Momentum-Anchored Ensemble (KAM-AE).
    Loads the test data, the trained ensemble models, and the optimized threshold to generate
    the final submission file.
    """

    @staticmethod
    def _get_feature_cols(df):
        """
        Identifies feature columns by excluding known metadata columns.
        """
        return [c for c in df.columns if c not in NON_FEATURE_COLS]

    @staticmethod
    def run_inference(load_cached_data=True):
        """
        Executes the full inference pipeline.

        Args:
            load_cached_data (bool): Whether to attempt loading cached test features.
        """
        seed_everything(SEED)
        print("\n[InferencePipeline] Starting Inference Pipeline...")

        # 1. Load Test Data
        # The DataLoader handles feature engineering and caching
        test_df = DataLoader.load_test_data(load_cached_data=load_cached_data)

        if test_df.empty:
            raise ValueError("Test data is empty. Cannot proceed with inference.")

        # Prepare Feature Matrix
        feature_cols = InferencePipeline._get_feature_cols(test_df)
        X_test = test_df[feature_cols]
        print(f" -> Test Data Shape: {X_test.shape}")

        # 2. Load Models
        model_dir = os.path.join(WORKING_DIR, "models")
        model_paths = [
            os.path.join(model_dir, "expert_lgbm.joblib"),
            os.path.join(model_dir, "expert_xgb.joblib"),
        ]

        # Verify models exist
        existing_models = [p for p in model_paths if os.path.exists(p)]
        if not existing_models:
            raise FileNotFoundError(
                f"No trained models found in {model_dir}. Please run training first."
            )

        predictor = EnsemblePredictor(existing_models)

        # 3. Load Optimal Threshold
        threshold_path = os.path.join(model_dir, "best_threshold.npy")
        if os.path.exists(threshold_path):
            best_threshold = np.load(threshold_path)[0]
            print(f" -> Loaded optimal threshold: {best_threshold}")
        else:
            best_threshold = 0.5
            print(
                f" -> Warning: Threshold file not found at {threshold_path}. Using default: 0.5"
            )

        # 4. Generate Predictions
        print(" -> Generating ensemble predictions...")
        # Get probabilities (average of models)
        y_pred_proba = predictor.predict_proba(X_test)

        # Apply threshold
        y_pred_binary = (y_pred_proba >= best_threshold).astype(int)

        # 5. Format Submission
        print(" -> Formatting submission...")
        submission = pd.DataFrame(
            {"contact_id": test_df["contact_id"], "contact": y_pred_binary}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"[InferencePipeline] Submission saved to {SUBMISSION_PATH}")
        print(f" -> Total predictions: {len(submission)}")
        print(f" -> Positive predictions: {submission['contact'].sum()}")
