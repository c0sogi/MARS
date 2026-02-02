import os
import pandas as pd
import numpy as np
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    LGBM_EXPERT_PARAMS,
    XGB_EXPERT_PARAMS,
)
from library.utils import setup_logger
from library.data_loader import DataLoader
from library.features import FeatureFactory
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper


class Predictor:
    """
    Manages the prediction pipeline for the test set.
    Generates Tier 2 features for the full test set, runs the Expert Ensemble,
    and formats the final submission.
    """

    def __init__(self, model_dir=WORKING_DIR):
        """
        Initialize the Predictor.

        Args:
            model_dir (str): Directory containing the saved model files.
        """
        self.logger = setup_logger()
        self.model_dir = model_dir
        self.loader = DataLoader()
        self.factory = FeatureFactory()
        self.lgbm_model = None
        self.xgb_model = None

    def load_models(self):
        """
        Loads the trained Expert models (LightGBM and XGBoost) from disk.
        """
        self.logger.info("Loading trained Expert models...")

        # Initialize wrappers
        # We pass the params to satisfy the constructor, though the actual model object
        # will be overwritten by the loaded joblib file.
        self.lgbm_model = LGBMClassifierWrapper(LGBM_EXPERT_PARAMS, name="expert_lgbm")
        self.xgb_model = XGBClassifierWrapper(XGB_EXPERT_PARAMS, name="expert_xgb")

        # Define paths
        lgbm_path = os.path.join(self.model_dir, "expert_lgbm.joblib")
        xgb_path = os.path.join(self.model_dir, "expert_xgb.joblib")

        # Load
        try:
            self.lgbm_model.load(lgbm_path)
            self.xgb_model.load(xgb_path)
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    def predict(self, threshold=0.5):
        """
        Executes the inference pipeline:
        1. Loads test data (base table).
        2. Computes Tier 2 features for the entire test set.
        3. Loads models (if not loaded).
        4. Predicts probabilities and ensembles them.
        5. Applies threshold and saves submission.

        Args:
            threshold (float): The decision threshold for classification.
                               Ideally obtained from the validation phase.
        """
        self.logger.info(f"Starting Inference Pipeline (Threshold={threshold})...")

        # 1. Load Test Data
        # The DataLoader handles reading metadata/tracking and merging them.
        # It also handles caching of the base table.
        df_test = self.loader.prepare_base_table(mode="test")

        # 2. Feature Generation
        # For the test set, we compute Tier 2 features for ALL rows.
        # The FeatureFactory handles caching of these features.
        self.logger.info("Computing Tier 2 features for the full test set...")
        X_test = self.factory.compute_tier2_features(df_test)

        # 3. Model Loading
        if self.lgbm_model is None or self.xgb_model is None:
            self.load_models()

        # 4. Inference
        self.logger.info("Running inference with Expert Ensemble...")

        # Get probabilities from both models
        prob_lgbm = self.lgbm_model.predict_proba(X_test)
        prob_xgb = self.xgb_model.predict_proba(X_test)

        # Ensemble: Simple Average
        prob_ensemble = 0.5 * prob_lgbm + 0.5 * prob_xgb

        # 5. Thresholding & Submission
        predictions = (prob_ensemble >= threshold).astype(int)

        self.logger.info(f"Generating submission file at {SUBMISSION_PATH}...")

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)

        # Log Statistics
        n_total = len(predictions)
        n_pos = predictions.sum()
        self.logger.info(f"Inference Complete.")
        self.logger.info(f"Total Predictions: {n_total}")
        self.logger.info(f"Positive Predictions: {n_pos} ({n_pos/n_total:.6f})")

        return submission_df
