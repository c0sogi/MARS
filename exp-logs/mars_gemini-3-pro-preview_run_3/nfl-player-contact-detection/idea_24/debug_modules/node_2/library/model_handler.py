import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger, get_config_hash


class DualStreamModel:
    """
    Manages the training, optimization, and inference of the Dual-Stream XGBoost architecture.
    Stream A: Interaction Model (Translational + Visual Consensus)
    Stream B: Impact Model (Rotational + Invariant)
    """

    def __init__(self):
        self.logger = setup_logger("DualStreamModel")
        self.config_hash = get_config_hash()
        self.working_dir = Config.WORKING_DIR

        # Paths for saving models
        self.model_a_path = os.path.join(
            self.working_dir, f"model_stream_a_{self.config_hash}.json"
        )
        self.model_b_path = os.path.join(
            self.working_dir, f"model_stream_b_{self.config_hash}.json"
        )

        # Thresholds (default, will be optimized)
        self.threshold_a = Config.DEFAULT_THRESHOLD
        self.threshold_b = Config.DEFAULT_THRESHOLD

        # In-memory models
        self.bst_a = None
        self.bst_b = None

    def train(self, train_data, val_data, force_retrain=False):
        """
        Trains both Stream A and Stream B models, and optimizes thresholds based on validation MCC.

        Args:
            train_data (dict): {'stream_a': {'X':..., 'y':...}, 'stream_b': ...}
            val_data (dict): {'stream_a': {'X':..., 'y':...}, 'stream_b': ...}
            force_retrain (bool): If True, ignores cached models and retrains.
        """
        self.logger.info("Starting Dual-Stream Training Pipeline...")

        # --- Train Stream A ---
        self.logger.info("--- Stream A: Interaction Model ---")
        self.bst_a = self._train_stream(
            train_data["stream_a"]["X"],
            train_data["stream_a"]["y"],
            val_data["stream_a"]["X"],
            val_data["stream_a"]["y"],
            Config.XGB_STREAM_A,
            self.model_a_path,
            force_retrain,
        )

        # --- Train Stream B ---
        self.logger.info("--- Stream B: Impact Model ---")
        self.bst_b = self._train_stream(
            train_data["stream_b"]["X"],
            train_data["stream_b"]["y"],
            val_data["stream_b"]["X"],
            val_data["stream_b"]["y"],
            Config.XGB_STREAM_B,
            self.model_b_path,
            force_retrain,
        )

        # --- Optimize Thresholds ---
        self.logger.info("Optimizing Decision Thresholds on Validation Set...")

        # Stream A Optimization
        if self.bst_a:
            dval_a = xgb.DMatrix(val_data["stream_a"]["X"])
            probs_a = self.bst_a.predict(dval_a)
            self.threshold_a, mcc_a = self._optimize_threshold(
                val_data["stream_a"]["y"], probs_a, "Stream A"
            )
            self.logger.info(
                f"Stream A Optimized: Threshold={self.threshold_a:.4f}, MCC={mcc_a:.6f}"
            )

        # Stream B Optimization
        if self.bst_b:
            dval_b = xgb.DMatrix(val_data["stream_b"]["X"])
            probs_b = self.bst_b.predict(dval_b)
            self.threshold_b, mcc_b = self._optimize_threshold(
                val_data["stream_b"]["y"], probs_b, "Stream B"
            )
            self.logger.info(
                f"Stream B Optimized: Threshold={self.threshold_b:.4f}, MCC={mcc_b:.6f}"
            )

    def predict(self, test_data):
        """
        Generates predictions for the test set using trained models and optimized thresholds.

        Args:
            test_data (dict): {'stream_a': {'X':..., 'ids':...}, 'stream_b': ...}

        Returns:
            pd.DataFrame: Submission dataframe with columns ['contact_id', 'contact']
        """
        self.logger.info("Generating predictions for Test Set...")

        # Ensure models are loaded
        if self.bst_a is None:
            if os.path.exists(self.model_a_path):
                self.bst_a = xgb.Booster()
                self.bst_a.load_model(self.model_a_path)
            else:
                self.logger.warning("Stream A model not found. Predictions will be 0.")

        if self.bst_b is None:
            if os.path.exists(self.model_b_path):
                self.bst_b = xgb.Booster()
                self.bst_b.load_model(self.model_b_path)
            else:
                self.logger.warning("Stream B model not found. Predictions will be 0.")

        results = []

        # --- Predict Stream A ---
        X_test_a = test_data["stream_a"]["X"]
        ids_test_a = test_data["stream_a"]["ids"]

        if not X_test_a.empty and self.bst_a:
            dtest_a = xgb.DMatrix(X_test_a)
            probs_a = self.bst_a.predict(dtest_a)
            preds_a = (probs_a >= self.threshold_a).astype(int)

            df_a = pd.DataFrame({"contact_id": ids_test_a, "contact": preds_a})
            results.append(df_a)
        elif not X_test_a.empty:
            # Fallback if model missing
            df_a = pd.DataFrame({"contact_id": ids_test_a, "contact": 0})
            results.append(df_a)

        # --- Predict Stream B ---
        X_test_b = test_data["stream_b"]["X"]
        ids_test_b = test_data["stream_b"]["ids"]

        if not X_test_b.empty and self.bst_b:
            dtest_b = xgb.DMatrix(X_test_b)
            probs_b = self.bst_b.predict(dtest_b)
            preds_b = (probs_b >= self.threshold_b).astype(int)

            df_b = pd.DataFrame({"contact_id": ids_test_b, "contact": preds_b})
            results.append(df_b)
        elif not X_test_b.empty:
            # Fallback if model missing
            df_b = pd.DataFrame({"contact_id": ids_test_b, "contact": 0})
            results.append(df_b)

        # --- Combine ---
        if results:
            submission = pd.concat(results, axis=0)
        else:
            submission = pd.DataFrame(columns=["contact_id", "contact"])

        # Ensure unique contact_ids (should be unique by design of streams)
        submission = submission.drop_duplicates(subset=["contact_id"])

        return submission

    def _train_stream(
        self, X_train, y_train, X_val, y_val, params, model_path, force_retrain
    ):
        """
        Internal helper to train a single stream with undersampling and early stopping.
        """
        # Check cache
        if not force_retrain and os.path.exists(model_path):
            self.logger.info(f"Loading cached model from {model_path}")
            bst = xgb.Booster()
            bst.load_model(model_path)
            return bst

        if X_train.empty:
            self.logger.warning(
                "Training data empty for this stream. Skipping training."
            )
            return None

        # Undersampling
        X_res, y_res = self._undersample(X_train, y_train)
        self.logger.info(
            f"Data after undersampling: {X_res.shape} (Positives: {np.sum(y_res)})"
        )

        # Create DMatrices
        dtrain = xgb.DMatrix(X_res, label=y_res)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        watchlist = [(dtrain, "train"), (dval, "eval")]

        # Suppress lightgbm/xgboost verbosity in output if needed, but we use verbose_eval
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=watchlist,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        # Save
        self.logger.info(f"Saving model to {model_path}")
        bst.save_model(model_path)

        return bst

    def _undersample(self, X, y):
        """
        Performs targeted majority undersampling.
        Keeps all positives. Samples negatives to achieve NEG_POS_RATIO.
        """
        # Convert to numpy for easier indexing if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X_np = X.values
            cols = X.columns
        else:
            X_np = X
            cols = None

        pos_mask = y == 1
        neg_mask = y == 0

        X_pos = X_np[pos_mask]
        y_pos = y[pos_mask]

        X_neg = X_np[neg_mask]
        y_neg = y[neg_mask]

        n_pos = len(y_pos)
        n_neg_keep = int(n_pos * Config.NEG_POS_RATIO)

        # If we have fewer negatives than the ratio, keep all
        if n_neg_keep < len(y_neg):
            indices = np.random.choice(len(y_neg), n_neg_keep, replace=False)
            X_neg_sampled = X_neg[indices]
            y_neg_sampled = y_neg[indices]
        else:
            X_neg_sampled = X_neg
            y_neg_sampled = y_neg

        # Concatenate
        X_combined = np.vstack([X_pos, X_neg_sampled])
        y_combined = np.concatenate([y_pos, y_neg_sampled])

        # Shuffle
        perm = np.random.permutation(len(y_combined))
        X_final = X_combined[perm]
        y_final = y_combined[perm]

        if cols is not None:
            X_final = pd.DataFrame(X_final, columns=cols)

        return X_final, y_final

    def _optimize_threshold(self, y_true, y_probs, stream_name):
        """
        Linear search for best MCC threshold.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search space: 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, preds)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc
