import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import setup_logger, set_seed
from library.feature_builder import FeatureBuilder


class DualStreamModel:
    """
    Implements the Context-Augmented Dual-Stream GBDT architecture.
    Manages training, threshold optimization, and inference for:
    - Stream A: Player-Player Interaction
    - Stream B: Player-Ground Impact (Context-Augmented)
    """

    def __init__(self):
        self.logger = setup_logger(name="DualStreamModel")
        self.feature_builder = FeatureBuilder()
        self.model_a = None
        self.model_b = None
        self.best_threshold_a = 0.5
        self.best_threshold_b = 0.5

    def train(self):
        """
        Orchestrates the training of both Stream A and Stream B models.
        """
        self.logger.info("Starting Dual-Stream Training Pipeline...")

        # --- Train Stream A (Player-Player) ---
        self.logger.info("\n=== Stream A: Player-Player Interaction Model ===")
        X_train_a, y_train_a, _ = self.feature_builder.generate_stream_a_features(
            "train"
        )
        X_val_a, y_val_a, _ = self.feature_builder.generate_stream_a_features(
            "validation"
        )

        self.model_a = self._train_stream(
            X_train_a,
            y_train_a,
            X_val_a,
            y_val_a,
            Config.XGB_PARAMS_STREAM_A,
            "StreamA",
        )

        # --- Train Stream B (Player-Ground) ---
        self.logger.info("\n=== Stream B: Player-Ground Impact Model ===")
        X_train_b, y_train_b, _ = self.feature_builder.generate_stream_b_features(
            "train"
        )
        X_val_b, y_val_b, _ = self.feature_builder.generate_stream_b_features(
            "validation"
        )

        self.model_b = self._train_stream(
            X_train_b,
            y_train_b,
            X_val_b,
            y_val_b,
            Config.XGB_PARAMS_STREAM_B,
            "StreamB",
        )

        self.logger.info("Training completed for both streams.")

    def _train_stream(self, X_train, y_train, X_val, y_val, params, stream_name):
        """
        Internal method to train a single XGBoost stream with undersampling and early stopping.
        """
        # 1. Random Undersampling of Training Data
        self.logger.info(
            f"[{stream_name}] Original Train Shape: {X_train.shape}, Positives: {np.sum(y_train)}"
        )

        pos_mask = y_train == 1
        neg_mask = y_train == 0

        X_pos = X_train[pos_mask]
        y_pos = y_train[pos_mask]
        X_neg = X_train[neg_mask]
        y_neg = y_train[neg_mask]

        n_pos = len(y_pos)
        n_neg_keep = int(n_pos * Config.UNDERSAMPLE_RATIO)

        # Sample negatives
        if n_neg_keep < len(y_neg):
            indices = np.random.choice(len(y_neg), n_neg_keep, replace=False)
            X_neg_sampled = X_neg.iloc[indices]
            y_neg_sampled = y_neg[indices]
        else:
            X_neg_sampled = X_neg
            y_neg_sampled = y_neg

        # Combine and Shuffle
        X_train_res = pd.concat([X_pos, X_neg_sampled])
        y_train_res = np.concatenate([y_pos, y_neg_sampled])

        # Shuffle
        shuffle_idx = np.random.permutation(len(y_train_res))
        X_train_res = X_train_res.iloc[shuffle_idx]
        y_train_res = y_train_res[shuffle_idx]

        self.logger.info(
            f"[{stream_name}] Resampled Train Shape: {X_train_res.shape}, Positives: {np.sum(y_train_res)}"
        )

        # 2. Prepare DMatrices
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 3. Train with Early Stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        return model

    def optimize_thresholds(self):
        """
        Optimizes decision thresholds for both streams using the validation set to maximize MCC.
        """
        self.logger.info("\n=== Optimizing Thresholds ===")

        # --- Optimize Stream A ---
        X_val_a, y_val_a, _ = self.feature_builder.generate_stream_a_features(
            "validation"
        )
        dval_a = xgb.DMatrix(X_val_a)
        probs_a = self.model_a.predict(dval_a)
        self.best_threshold_a, mcc_a = self._find_best_threshold(
            y_val_a, probs_a, "StreamA"
        )

        # --- Optimize Stream B ---
        X_val_b, y_val_b, _ = self.feature_builder.generate_stream_b_features(
            "validation"
        )
        dval_b = xgb.DMatrix(X_val_b)
        probs_b = self.model_b.predict(dval_b)
        self.best_threshold_b, mcc_b = self._find_best_threshold(
            y_val_b, probs_b, "StreamB"
        )

        self.logger.info(
            f"Optimization Complete. Best Thresh A: {self.best_threshold_a:.4f}, Best Thresh B: {self.best_threshold_b:.4f}"
        )

    def _find_best_threshold(self, y_true, y_probs, stream_name):
        """
        Linear search for optimal MCC threshold.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range: 0.01 to 0.99
        thresholds = np.arange(0.01, 1.00, 0.01)

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        self.logger.info(
            f"[{stream_name}] Best MCC: {best_mcc} at Threshold: {best_thresh:.2f}"
        )
        return best_thresh, best_mcc

    def generate_submission(self):
        """
        Generates predictions for the test set and creates the submission file.
        """
        self.logger.info("\n=== Generating Submission ===")

        # --- Predict Stream A (Test) ---
        X_test_a, _, ids_a = self.feature_builder.generate_stream_a_features("test")
        dtest_a = xgb.DMatrix(X_test_a)
        probs_a = self.model_a.predict(dtest_a)
        preds_a = (probs_a >= self.best_threshold_a).astype(int)

        df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})

        # --- Predict Stream B (Test) ---
        X_test_b, _, ids_b = self.feature_builder.generate_stream_b_features("test")
        dtest_b = xgb.DMatrix(X_test_b)
        probs_b = self.model_b.predict(dtest_b)
        preds_b = (probs_b >= self.best_threshold_b).astype(int)

        df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})

        # --- Combine and Format ---
        df_submission = pd.concat([df_a, df_b], ignore_index=True)

        # Ensure we match the sample submission exactly
        # Load sample submission to get the exact order and list of IDs
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission structure
        # We use left join on sample_sub to ensure we have all rows and correct order
        final_sub = sample_sub[["contact_id"]].merge(
            df_submission, on="contact_id", how="left"
        )

        # Fill missing values with 0 (default no contact) just in case
        missing_count = final_sub["contact"].isna().sum()
        if missing_count > 0:
            self.logger.warning(
                f"Found {missing_count} missing predictions. Filling with 0."
            )
            final_sub["contact"] = final_sub["contact"].fillna(0)

        final_sub["contact"] = final_sub["contact"].astype(int)

        # Save
        final_sub.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)
        self.logger.info(
            f"Submission saved to {Config.OUTPUT_SUBMISSION_PATH}. Rows: {len(final_sub)}"
        )
