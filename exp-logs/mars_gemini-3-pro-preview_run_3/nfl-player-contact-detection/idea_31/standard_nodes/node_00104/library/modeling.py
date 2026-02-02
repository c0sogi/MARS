import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger
from library.data_pipeline import DataPipeline


class DualStreamModel:
    """
    Implements the Physically-Consistent Hybrid-Context Dual-Stream GBDT.
    Manages training, threshold optimization, and inference for Stream A (Interaction)
    and Stream B (Impact) models.
    """

    def __init__(self):
        self.logger = setup_logger("DualStreamModel")
        self.pipeline = DataPipeline()
        self.model_a = None
        self.model_b = None
        self.threshold_a = 0.5
        self.threshold_b = 0.5

        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path_a = os.path.join(self.model_dir, "xgb_stream_a.json")
        self.model_path_b = os.path.join(self.model_dir, "xgb_stream_b.json")

    def _train_stream(self, stream_name, params, X_train, y_train, X_val, y_val):
        """
        Generic internal method to train an XGBoost model for a specific stream.
        """
        self.logger.info(f"Training {stream_name}...")
        self.logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        return model

    def _optimize_threshold(self, y_true, y_prob, stream_name):
        """
        Finds the probability threshold that maximizes MCC.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Search range: 0.1 to 0.9
        thresholds = np.linspace(0.1, 0.9, 81)

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        self.logger.info(
            f"Best Threshold for {stream_name}: {best_threshold:.4f} (MCC: {best_mcc})"
        )
        return best_threshold

    def train(self):
        """
        Orchestrates the training process for both streams.
        1. Load Data (Train & Val)
        2. Train Models
        3. Optimize Thresholds
        4. Save Models
        """
        self.logger.info("Starting Dual-Stream Training...")

        # ==========================================
        # Stream A: Interaction Model (Player-Player)
        # ==========================================
        X_train_a, y_train_a, _ = self.pipeline.load_data(
            mode="train", stream="streamA"
        )
        X_val_a, y_val_a, _ = self.pipeline.load_data(
            mode="validation", stream="streamA"
        )

        self.model_a = self._train_stream(
            "Stream A", Config.STREAM_A_PARAMS, X_train_a, y_train_a, X_val_a, y_val_a
        )

        # Predict on validation to find optimal threshold
        dval_a = xgb.DMatrix(X_val_a)
        y_prob_a = self.model_a.predict(dval_a)
        self.threshold_a = self._optimize_threshold(y_val_a, y_prob_a, "Stream A")

        # Free memory
        del X_train_a, y_train_a, X_val_a, y_val_a, dval_a
        import gc

        gc.collect()

        # ==========================================
        # Stream B: Impact Model (Player-Ground)
        # ==========================================
        X_train_b, y_train_b, _ = self.pipeline.load_data(
            mode="train", stream="streamB"
        )
        X_val_b, y_val_b, _ = self.pipeline.load_data(
            mode="validation", stream="streamB"
        )

        self.model_b = self._train_stream(
            "Stream B", Config.STREAM_B_PARAMS, X_train_b, y_train_b, X_val_b, y_val_b
        )

        dval_b = xgb.DMatrix(X_val_b)
        y_prob_b = self.model_b.predict(dval_b)
        self.threshold_b = self._optimize_threshold(y_val_b, y_prob_b, "Stream B")

        # Free memory
        del X_train_b, y_train_b, X_val_b, y_val_b, dval_b
        gc.collect()

        # Save Models
        self.model_a.save_model(self.model_path_a)
        self.model_b.save_model(self.model_path_b)

        # Save Thresholds
        thresholds = {"streamA": self.threshold_a, "streamB": self.threshold_b}
        joblib.dump(thresholds, os.path.join(self.model_dir, "thresholds.joblib"))

        self.logger.info("Training completed successfully.")

    def predict(self):
        """
        Generates predictions for the test set.
        1. Load Test Data for both streams
        2. Predict using trained models
        3. Apply optimized thresholds
        4. Merge and save submission
        """
        self.logger.info("Starting Inference...")

        # Ensure models are loaded
        if self.model_a is None or self.model_b is None:
            if os.path.exists(self.model_path_a) and os.path.exists(self.model_path_b):
                self.logger.info("Loading models from disk...")
                self.model_a = xgb.Booster()
                self.model_a.load_model(self.model_path_a)
                self.model_b = xgb.Booster()
                self.model_b.load_model(self.model_path_b)

                thresh_path = os.path.join(self.model_dir, "thresholds.joblib")
                if os.path.exists(thresh_path):
                    thresholds = joblib.load(thresh_path)
                    self.threshold_a = thresholds["streamA"]
                    self.threshold_b = thresholds["streamB"]
            else:
                raise RuntimeError("Models not found. Run train() first.")

        # Dictionary to store results: contact_id -> prediction (int)
        results = {}

        # ==========================================
        # Stream A Inference
        # ==========================================
        X_test_a, _, ids_a = self.pipeline.load_data(mode="test", stream="streamA")
        if len(ids_a) > 0:
            dtest_a = xgb.DMatrix(X_test_a)
            probs_a = self.model_a.predict(dtest_a)
            preds_a = (probs_a >= self.threshold_a).astype(int)

            for cid, pred in zip(ids_a, preds_a):
                results[cid] = pred

            del X_test_a, dtest_a, probs_a
            import gc

            gc.collect()

        # ==========================================
        # Stream B Inference
        # ==========================================
        X_test_b, _, ids_b = self.pipeline.load_data(mode="test", stream="streamB")
        if len(ids_b) > 0:
            dtest_b = xgb.DMatrix(X_test_b)
            probs_b = self.model_b.predict(dtest_b)
            preds_b = (probs_b >= self.threshold_b).astype(int)

            for cid, pred in zip(ids_b, preds_b):
                results[cid] = pred

            del X_test_b, dtest_b, probs_b
            gc.collect()

        # ==========================================
        # Submission Generation
        # ==========================================
        self.logger.info("Generating submission file...")

        # Load sample submission to ensure correct order and completeness
        df_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Map predictions
        # Default to 0 if not found (though all should be covered by the two streams)
        df_sub["contact"] = df_sub["contact_id"].map(results).fillna(0).astype(int)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Positive predictions count: {df_sub['contact'].sum()}")
