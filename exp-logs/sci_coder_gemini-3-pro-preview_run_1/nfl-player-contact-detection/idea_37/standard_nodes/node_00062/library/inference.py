import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger
from library.data_processing import DataProcessor
from library.models import LGBMWrapper, XGBWrapper


class InferenceManager:
    def __init__(self):
        self.logger = setup_logger("inference")
        self.processor = DataProcessor()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.submission_dir = Config.SUBMISSION_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Columns to exclude from feature set during prediction
        self.meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]

    def load_models(self):
        """
        Loads the trained Expert LightGBM and XGBoost models from disk.
        """
        lgbm_path = os.path.join(self.models_dir, "expert_lgbm.joblib")
        xgb_path = os.path.join(self.models_dir, "expert_xgb.joblib")

        if not os.path.exists(lgbm_path) or not os.path.exists(xgb_path):
            raise FileNotFoundError("Trained models not found. Run training first.")

        expert_lgbm = LGBMWrapper()
        expert_lgbm.load(lgbm_path)

        expert_xgb = XGBWrapper()
        expert_xgb.load(xgb_path)

        return expert_lgbm, expert_xgb

    def load_threshold(self):
        """
        Loads the optimized threshold from disk. Returns 0.5 if not found.
        """
        thresh_path = os.path.join(self.models_dir, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = np.load(thresh_path)[0]
            self.logger.info(f"Loaded optimized threshold: {threshold:.16f}")
            return threshold
        else:
            self.logger.warning("Optimized threshold not found. Using default 0.5.")
            return 0.5

    def generate_test_features(self, load_cached_data=True):
        """
        Generates or loads features for the test set using DataProcessor.
        """
        self.logger.info("Generating test features...")
        return self.processor.get_test_data(
            load_cached=load_cached_data, debug=Config.DEBUG
        )

    def optimize_threshold(self, df_val, expert_lgbm, expert_xgb):
        """
        Optimizes the decision threshold on the validation set to maximize MCC.
        Useful if we want to re-evaluate without full training.
        """
        self.logger.info("Optimizing threshold on validation set...")

        feature_cols = [c for c in df_val.columns if c not in self.meta_cols]
        X_val = df_val[feature_cols]
        y_val = df_val["contact"].values

        # Generate predictions
        p_lgbm = expert_lgbm.predict(X_val)
        p_xgb = expert_xgb.predict(X_val)

        # Ensemble
        w_lgbm = Config.ENSEMBLE_WEIGHTS["lgbm"]
        w_xgb = Config.ENSEMBLE_WEIGHTS["xgb"]
        p_ens = (p_lgbm * w_lgbm) + (p_xgb * w_xgb)

        # Sweep
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (p_ens > t).astype(int)
            mcc = matthews_corrcoef(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        self.logger.info(
            f"Validation MCC: {best_mcc:.16f} at Threshold: {best_thresh:.16f}"
        )

        # Save for consistency
        np.save(
            os.path.join(self.models_dir, "best_threshold.npy"), np.array([best_thresh])
        )

        return best_thresh

    def run_ensemble_inference(self, df_test, expert_lgbm, expert_xgb, threshold):
        """
        Runs inference on the test set using the ensemble of models and applies the threshold.
        """
        self.logger.info(f"Running ensemble inference on {len(df_test)} samples...")

        feature_cols = [c for c in df_test.columns if c not in self.meta_cols]
        X_test = df_test[feature_cols]

        # Predict
        p_lgbm = expert_lgbm.predict(X_test)
        p_xgb = expert_xgb.predict(X_test)

        # Weighted Average
        w_lgbm = Config.ENSEMBLE_WEIGHTS["lgbm"]
        w_xgb = Config.ENSEMBLE_WEIGHTS["xgb"]

        probs = (p_lgbm * w_lgbm) + (p_xgb * w_xgb)

        # Apply Threshold
        predictions = (probs > threshold).astype(int)

        return predictions

    def generate_submission(
        self, load_cached_data=True, run_validation_optimization=False
    ):
        """
        Main driver to generate the submission file.

        Args:
            load_cached_data: Whether to use cached feature files.
            run_validation_optimization: If True, loads validation data to re-optimize threshold.
        """
        # 1. Load Models
        expert_lgbm, expert_xgb = self.load_models()

        # 2. Determine Threshold
        if run_validation_optimization:
            df_val = self.processor.get_val_data(
                load_cached=load_cached_data, debug=Config.DEBUG
            )
            threshold = self.optimize_threshold(df_val, expert_lgbm, expert_xgb)
        else:
            threshold = self.load_threshold()

        # 3. Load Test Data
        df_test = self.generate_test_features(load_cached_data=load_cached_data)

        if df_test.empty:
            self.logger.warning("Test data is empty. Generating empty submission.")
            # Handle edge case or create dummy submission based on sample
            # For this context, we assume data exists.
            return

        # 4. Run Inference
        preds = self.run_ensemble_inference(df_test, expert_lgbm, expert_xgb, threshold)

        # 5. Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": preds}
        )

        # 6. Save
        os.makedirs(self.submission_dir, exist_ok=True)
        submission.to_csv(self.submission_path, index=False)
        self.logger.info(f"Submission saved to {self.submission_path}")
        self.logger.info(f"Submission shape: {submission.shape}")
        self.logger.info(f"Predicted positive contacts: {submission['contact'].sum()}")
