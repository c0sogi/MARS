import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger
from library.data_pipeline import DataPipeline


class InferenceEngine:
    """
    Manages prediction and threshold optimization for the Dual-Stream Contact Detection model.
    """

    def __init__(self):
        self.logger = setup_logger("InferenceEngine")
        self.pipeline = DataPipeline()

        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        self.model_path_a = os.path.join(self.model_dir, "xgb_stream_a.json")
        self.model_path_b = os.path.join(self.model_dir, "xgb_stream_b.json")
        self.threshold_path = os.path.join(self.model_dir, "thresholds.joblib")

    def _load_model(self, model_path):
        """Loads an XGBoost model from a JSON file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster

    def optimize_thresholds(self, y_true, y_prob, stream_name):
        """
        Performs a linear search to find the probability threshold that maximizes MCC.

        Args:
            y_true (np.array): Ground truth labels.
            y_prob (np.array): Predicted probabilities.
            stream_name (str): Name of the stream for logging.

        Returns:
            float: The optimal threshold.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Search range: 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        self.logger.info(f"Optimization Results for {stream_name}:")
        self.logger.info(f"Best Threshold: {best_threshold}")
        self.logger.info(f"Best MCC: {best_mcc}")

        return best_threshold

    def predict(self, use_validation=True):
        """
        Generates predictions for the test set.

        Args:
            use_validation (bool): If True, loads validation data to optimize thresholds.
                                   If False, attempts to load pre-calculated thresholds.
        """
        self.logger.info("Starting Inference Pipeline...")

        # Load Models
        self.logger.info("Loading models...")
        model_a = self._load_model(self.model_path_a)
        model_b = self._load_model(self.model_path_b)

        threshold_a = 0.5
        threshold_b = 0.5

        # ---------------------------------------------------------
        # 1. Threshold Optimization (Optional but recommended)
        # ---------------------------------------------------------
        if use_validation:
            self.logger.info("Optimizing thresholds using validation set...")

            # Stream A
            X_val_a, y_val_a, _ = self.pipeline.load_data(
                mode="validation", stream="streamA"
            )
            if len(y_val_a) > 0:
                dval_a = xgb.DMatrix(X_val_a)
                probs_a = model_a.predict(dval_a)
                threshold_a = self.optimize_thresholds(y_val_a, probs_a, "Stream A")
                del X_val_a, dval_a, probs_a

            # Stream B
            X_val_b, y_val_b, _ = self.pipeline.load_data(
                mode="validation", stream="streamB"
            )
            if len(y_val_b) > 0:
                dval_b = xgb.DMatrix(X_val_b)
                probs_b = model_b.predict(dval_b)
                threshold_b = self.optimize_thresholds(y_val_b, probs_b, "Stream B")
                del X_val_b, dval_b, probs_b

            # Save thresholds for future reference
            joblib.dump(
                {"streamA": threshold_a, "streamB": threshold_b}, self.threshold_path
            )

        else:
            self.logger.info("Loading thresholds from disk...")
            if os.path.exists(self.threshold_path):
                thresholds = joblib.load(self.threshold_path)
                threshold_a = thresholds.get("streamA", 0.5)
                threshold_b = thresholds.get("streamB", 0.5)
            else:
                self.logger.warning("Threshold file not found. Using default 0.5.")

        self.logger.info(
            f"Using Thresholds - Stream A: {threshold_a}, Stream B: {threshold_b}"
        )

        # ---------------------------------------------------------
        # 2. Test Set Prediction
        # ---------------------------------------------------------
        results = {}

        # Stream A Inference
        self.logger.info("Predicting Stream A (Interaction)...")
        X_test_a, _, ids_a = self.pipeline.load_data(mode="test", stream="streamA")
        if len(ids_a) > 0:
            dtest_a = xgb.DMatrix(X_test_a)
            probs_a = model_a.predict(dtest_a)
            preds_a = (probs_a >= threshold_a).astype(int)

            for cid, pred in zip(ids_a, preds_a):
                results[cid] = pred

            del X_test_a, dtest_a, probs_a

        # Stream B Inference
        self.logger.info("Predicting Stream B (Impact)...")
        X_test_b, _, ids_b = self.pipeline.load_data(mode="test", stream="streamB")
        if len(ids_b) > 0:
            dtest_b = xgb.DMatrix(X_test_b)
            probs_b = model_b.predict(dtest_b)
            preds_b = (probs_b >= threshold_b).astype(int)

            for cid, pred in zip(ids_b, preds_b):
                results[cid] = pred

            del X_test_b, dtest_b, probs_b

        # ---------------------------------------------------------
        # 3. Submission Generation
        # ---------------------------------------------------------
        self.logger.info("Generating submission file...")

        # Load sample submission to ensure structure
        df_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Map results
        # Fill missing with 0 (though pipeline should cover all valid IDs)
        df_sub["contact"] = df_sub["contact_id"].map(results).fillna(0).astype(int)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Total Positive Predictions: {df_sub['contact'].sum()}")
