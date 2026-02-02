import os
import joblib
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, garbage_collection
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.model_factory import ModelFactory


class InferencePipeline:
    """
    Manages the inference workflow for the VRC-ME architecture.
    Loads trained Expert models, generates high-resolution features for the test set,
    performs ensemble prediction, and generates the final submission file.
    """

    def __init__(self):
        self.logger = setup_logger(name="InferencePipeline")
        self.loader = DataLoader()
        self.generator = FeatureGenerator()

        # Artifact paths (must match those used in ExpertTrainer)
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        self.lgbm_path = os.path.join(self.model_dir, "lgbm_expert.joblib")
        self.xgb_path = os.path.join(self.model_dir, "xgb_expert.joblib")
        self.threshold_path = os.path.join(self.model_dir, "threshold.joblib")

        # Output path
        self.submission_path = Config.SUBMISSION_FILE

    def _get_features(self, df: pd.DataFrame):
        """
        Extracts Tier 2 features from the dataframe.
        """
        feature_cols = [c for c in Config.TIER2_FEATURES if c in df.columns]
        return df[feature_cols]

    def run(self, load_cached_data: bool = True):
        """
        Executes the full inference pipeline.

        Args:
            load_cached_data (bool): Whether to use cached feature datasets.
        """
        self.logger.info("Starting Inference Pipeline...")

        # ---------------------------------------------------------
        # 1. Load Artifacts
        # ---------------------------------------------------------
        self.logger.info("Loading model artifacts...")

        if not os.path.exists(self.lgbm_path) or not os.path.exists(self.xgb_path):
            raise FileNotFoundError(
                f"Models not found in {self.model_dir}. Ensure training is complete."
            )

        if not os.path.exists(self.threshold_path):
            raise FileNotFoundError(
                f"Threshold file not found at {self.threshold_path}."
            )

        # Load Models
        lgbm_model = ModelFactory.create_model(stage="expert", model_type="lgbm")
        lgbm_model.load(self.lgbm_path)

        xgb_model = ModelFactory.create_model(stage="expert", model_type="xgb")
        xgb_model.load(self.xgb_path)

        # Load Threshold
        threshold = joblib.load(self.threshold_path)
        self.logger.info(f"Loaded optimized decision threshold: {threshold}")

        # ---------------------------------------------------------
        # 2. Data Loading & Feature Generation
        # ---------------------------------------------------------
        self.logger.info("Loading Test Data...")
        merged_test = self.loader.get_merged_data(
            split="test", load_cached_data=load_cached_data
        )
        tracking_test = self.loader.load_tracking(split="test")

        self.logger.info("Generating Tier 2 features for Test set...")
        # We use Tier 2 features (full context) for the test set since it's small enough
        df_test_tier2 = self.generator.generate(
            merged_test,
            tracking_test,
            tier=2,
            split="test",
            load_cached_data=load_cached_data,
        )

        # Free memory of raw data
        del merged_test, tracking_test
        garbage_collection()

        # ---------------------------------------------------------
        # 3. Ensemble Prediction
        # ---------------------------------------------------------
        self.logger.info("Running Ensemble Inference...")

        X_test = self._get_features(df_test_tier2)

        self.logger.info(
            f"Predicting on {len(X_test)} samples with {X_test.shape[1]} features."
        )

        # LightGBM Prediction
        p_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

        # XGBoost Prediction
        p_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # Average Ensemble
        p_ensemble = (p_lgbm + p_xgb) / 2.0

        # Apply Threshold
        predictions = (p_ensemble >= threshold).astype(int)

        # ---------------------------------------------------------
        # 4. Generate Submission
        # ---------------------------------------------------------
        self.logger.info("Generating submission file...")

        # Ensure we have the contact_id column
        if "contact_id" not in df_test_tier2.columns:
            raise ValueError("contact_id column missing from feature dataframe.")

        submission = df_test_tier2[["contact_id"]].copy()
        submission["contact"] = predictions

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)

        submission.to_csv(self.submission_path, index=False)

        self.logger.info(f"Submission saved to {self.submission_path}")
        self.logger.info(f"Final Submission Shape: {submission.shape}")

        # Cleanup
        del df_test_tier2, X_test, p_lgbm, p_xgb, p_ensemble, predictions, submission
        garbage_collection()
