import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from typing import Tuple, Dict, Any

from library.config import Config
from library.utils import setup_logging, CacheManager
from library.data_processing import DataLoader
from library.feature_engineering import KinematicFeatureEngine
from library.models import LGBMWrapper, XGBWrapper


class InferenceEngine:
    """
    Manages the inference pipeline for KARP-AM.
    1. Loads trained Expert models.
    2. Optimizes decision threshold using the Validation set.
    3. Generates predictions for the Test set (handling Gating logic).
    4. Formats and saves the final submission.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.cache = CacheManager()
        self.data_loader = DataLoader()
        self.feature_engine = KinematicFeatureEngine()
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")

        # Placeholders for models
        self.model_lgbm = None
        self.model_xgb = None

    def run_inference(self, load_cached_data: bool = True):
        """
        Main execution method for the inference pipeline.
        """
        self.logger.info("Starting Inference Pipeline...")

        # 1. Load Models
        self._load_models()

        # 2. Optimize Threshold on Validation Set
        # We re-calculate this to ensure the threshold is optimized for the ENSEMBLE,
        # not just individual models.
        best_threshold = self._optimize_threshold(load_cached_data=load_cached_data)
        self.logger.info(f"Optimal Ensemble Threshold determined: {best_threshold}")

        # 3. Generate Test Predictions
        predictions_df = self._predict_test_set(load_cached_data=load_cached_data)

        # 4. Create and Save Submission
        self._create_submission(predictions_df, best_threshold)

        self.logger.info("Inference Pipeline Completed Successfully.")

    def _load_models(self):
        """
        Loads the trained Expert models from disk.
        """
        self.logger.info("Loading trained models...")

        self.model_lgbm = LGBMWrapper(name="expert_lgbm")
        self.model_lgbm.load(self.model_dir)

        self.model_xgb = XGBWrapper(name="expert_xgb")
        self.model_xgb.load(self.model_dir)

    def _optimize_threshold(self, load_cached_data: bool) -> float:
        """
        Loads validation data, generates ensemble predictions, and finds the threshold
        that maximizes MCC.
        """
        self.logger.info("Optimizing threshold using Validation set...")

        # Load Val Data
        meta_val, track_val = self.data_loader.load_data(
            "val", load_cached_data=load_cached_data
        )

        # Generate Val Features
        df_val = self.feature_engine.process_data(
            meta_val, track_val, dataset_key="val", load_cached_data=load_cached_data
        )

        # Prepare X, y
        X_val, y_val = self._prepare_features(df_val)

        # Predict
        p_lgbm = self.model_lgbm.predict_proba(X_val)
        p_xgb = self.model_xgb.predict_proba(X_val)

        # Ensemble Average
        p_ensemble = (p_lgbm + p_xgb) / 2.0

        # Grid Search for Best Threshold
        thresholds = np.linspace(0.1, 0.9, 801)  # High resolution
        best_mcc = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (p_ensemble >= t).astype(int)
            score = matthews_corrcoef(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        self.logger.info(f"Validation Best MCC: {best_mcc} at Threshold: {best_thresh}")
        return best_thresh

    def _predict_test_set(self, load_cached_data: bool) -> pd.DataFrame:
        """
        Loads test data, generates features, and predicts probabilities.
        Returns a DataFrame with ['contact_id', 'probability'].
        """
        self.logger.info("Processing Test Set...")

        # Load Test Data
        meta_test, track_test = self.data_loader.load_data(
            "test", load_cached_data=load_cached_data
        )

        # Generate Test Features
        # Note: The Feature Engine applies Gating.
        # Rows dropped here are "Easy Negatives" and won't be in df_test.
        df_test = self.feature_engine.process_data(
            meta_test, track_test, dataset_key="test", load_cached_data=load_cached_data
        )

        if df_test.empty:
            self.logger.warning(
                "Test set features are empty after gating. Returning empty predictions."
            )
            return pd.DataFrame(columns=["contact_id", "probability"])

        # Prepare X (No y in test)
        X_test, _ = self._prepare_features(df_test, is_test=True)

        self.logger.info(f"Predicting on {len(df_test)} surviving test pairs...")

        # Predict
        p_lgbm = self.model_lgbm.predict_proba(X_test)
        p_xgb = self.model_xgb.predict_proba(X_test)

        # Ensemble
        p_ensemble = (p_lgbm + p_xgb) / 2.0

        # Create Result DataFrame
        results = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "probability": p_ensemble}
        )

        return results

    def _prepare_features(
        self, df: pd.DataFrame, is_test: bool = False
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Strips metadata columns to prepare feature matrix X.
        """
        drop_cols = [
            "contact_id",
            "game_play",
            "step",
            "contact",
            "datetime",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols]

        y = None
        if not is_test and "contact" in df.columns:
            y = df["contact"]

        return X, y

    def _create_submission(self, predictions_df: pd.DataFrame, threshold: float):
        """
        Merges predictions with the sample submission template, handles missing (gated) rows,
        applies threshold, and saves to CSV.
        """
        self.logger.info("Generating submission file...")

        # Load Sample Submission to get the exhaustive list of contact_ids
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # We only need contact_id from sample submission
        # Ensure we don't have 'contact' col conflict if it exists in sample
        if "contact" in sample_sub.columns:
            template = sample_sub[["contact_id"]].copy()
        else:
            template = sample_sub.copy()

        # Merge Predictions
        # Left join: Keep all rows from template.
        # Rows that were filtered out by Gating will have NaN in 'probability'.
        submission = template.merge(predictions_df, on="contact_id", how="left")

        # Fill NaNs with 0.0 (Easy Negatives filtered by Gating)
        n_gated = submission["probability"].isna().sum()
        self.logger.info(f"Imputing {n_gated} gated-out pairs as No Contact (0.0).")
        submission["probability"] = submission["probability"].fillna(0.0)

        # Apply Threshold
        submission["contact"] = (submission["probability"] >= threshold).astype(int)

        # Select final columns
        final_submission = submission[["contact_id", "contact"]]

        # Save
        save_path = Config.SUBMISSION_FILE
        final_submission.to_csv(save_path, index=False)
        self.logger.info(f"Submission saved to {save_path}")
        self.logger.info(f"Submission shape: {final_submission.shape}")

        # Quick sanity check
        pos_count = final_submission["contact"].sum()
        self.logger.info(
            f"Predicted Positives: {pos_count} ({pos_count/len(final_submission):.4f})"
        )
