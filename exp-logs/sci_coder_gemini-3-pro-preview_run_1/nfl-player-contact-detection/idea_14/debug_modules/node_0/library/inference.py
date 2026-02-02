import os
import numpy as np
import pandas as pd
from typing import Optional

from library.config import MODEL_OUTPUT_DIR, SUBMISSION_PATH, SEED, FeatureConfig
from library.data_factory import DataFactory
from library.model_factory import EnsemblePredictor
from library.utils import set_seed


class InferencePipeline:
    """
    Manages the inference process: generating test features, loading the expert ensemble,
    applying the optimized threshold, and creating the submission file.
    """

    def __init__(self):
        set_seed(SEED)
        self.feature_config = FeatureConfig()
        self.data_factory = DataFactory(self.feature_config)

        # Define paths for artifacts
        self.lgbm_path = os.path.join(MODEL_OUTPUT_DIR, "expert_lgbm.joblib")
        self.xgb_path = os.path.join(MODEL_OUTPUT_DIR, "expert_xgb.joblib")
        self.threshold_path = os.path.join(MODEL_OUTPUT_DIR, "best_threshold.npy")

    def _load_threshold(self) -> float:
        """
        Loads the optimized threshold from disk. Defaults to 0.5 if not found.
        """
        if os.path.exists(self.threshold_path):
            try:
                threshold = float(np.load(self.threshold_path)[0])
                print(f"Loaded optimized threshold: {threshold:.16f}")
                return threshold
            except Exception as e:
                print(f"Error loading threshold file: {e}. Defaulting to 0.5.")
                return 0.5
        else:
            print("Threshold file not found. Defaulting to 0.5.")
            return 0.5

    def generate_submission(self, load_cached_data: bool = True) -> None:
        """
        Executes the full inference workflow and saves the submission.csv.

        Args:
            load_cached_data (bool): Whether to use cached test features if available.
        """
        print("Starting Inference Pipeline...")

        # 1. Generate/Load Test Features
        # Note: Gating is NOT applied in test mode (mode='test') inside DataFactory,
        # ensuring we predict for all rows in sample_submission.
        print("Retrieving test dataset features...")
        df_test = self.data_factory.get_processed_dataset(
            mode="test", load_cached_data=load_cached_data
        )

        # 2. Prepare Feature Matrix (X)
        # Identify columns to drop to isolate the feature set
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]

        # Select numeric features only
        feature_cols = [c for c in df_test.columns if c not in meta_cols]
        X_test = df_test[feature_cols]

        print(f"Test Feature Matrix Shape: {X_test.shape}")

        # 3. Load Models and Predict
        print("Loading Expert Ensemble models...")

        # Check existence of model files to pass correct paths to EnsemblePredictor
        lgbm_file = self.lgbm_path if os.path.exists(self.lgbm_path) else None
        xgb_file = self.xgb_path if os.path.exists(self.xgb_path) else None

        if lgbm_file is None and xgb_file is None:
            raise RuntimeError(
                f"No trained models found in {MODEL_OUTPUT_DIR}. Cannot proceed with inference."
            )

        predictor = EnsemblePredictor(lgbm_path=lgbm_file, xgb_path=xgb_file)

        print("Running inference...")
        probs = predictor.predict(X_test)

        # 4. Apply Threshold
        threshold = self._load_threshold()
        preds_binary = (probs > threshold).astype(int)

        # 5. Create Submission DataFrame
        # We rely on df_test['contact_id'] which comes from test_metadata.csv (derived from sample_submission)
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": preds_binary}
        )

        # 6. Save Submission
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)

        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Total rows: {len(submission)}")
        print(
            f"Predicted Contacts: {submission['contact'].sum()} ({(submission['contact'].mean() * 100):.2f}%)"
        )
        print("Inference Pipeline completed successfully.")
