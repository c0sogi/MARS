import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from sklearn.metrics import matthews_corrcoef
from typing import Dict, Tuple, Any

from library.config import (
    XGB_PARAMS_STREAM_A,
    XGB_PARAMS_STREAM_B,
    NEG_POS_RATIO,
    SEED,
    EARLY_STOPPING_ROUNDS,
    WORKING_DIR,
    SUBMISSION_DIR,
    SAMPLE_SUBMISSION_PATH,
)
from library.utils import seed_everything


class ModelTrainer:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.models_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        seed_everything(SEED)

    def undersample_majority(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Performs targeted majority undersampling.
        Retains 100% of positive class and subsamples negative class to NEG_POS_RATIO:1.
        """
        # Combine for easier filtering
        df = X.copy()
        df["target_temp"] = y

        pos_mask = df["target_temp"] == 1
        neg_mask = df["target_temp"] == 0

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_pos == 0:
            print(
                "Warning: No positive samples found in training data. Skipping undersampling."
            )
            return X, y

        target_n_neg = int(n_pos * NEG_POS_RATIO)

        if target_n_neg >= n_neg:
            # No need to undersample if we already have fewer negatives than the ratio
            return X, y

        # Sample negatives
        df_pos = df[pos_mask]
        df_neg = df[neg_mask].sample(n=target_n_neg, random_state=SEED)

        # Combine and shuffle
        df_resampled = (
            pd.concat([df_pos, df_neg])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )

        y_resampled = df_resampled["target_temp"].values
        X_resampled = df_resampled.drop(columns=["target_temp"])

        return X_resampled, y_resampled

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        params: Dict[str, Any],
        model_name: str,
    ) -> xgb.Booster:
        """
        Trains an XGBoost model with early stopping.
        """
        print(f"Training {model_name}...")

        # Undersample training data
        X_train_res, y_train_res = self.undersample_majority(X_train, y_train)
        print(
            f"  Train shape after undersampling: {X_train_res.shape}, Positive rate: {y_train_res.mean():.4f}"
        )

        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators", 1000),
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}.json")
        model.save_model(model_path)
        print(f"  Model saved to {model_path}")

        # Log best score
        print(f"  Best Iteration: {model.best_iteration}")
        print(f"  Best Score (LogLoss): {model.best_score}")

        return model

    def optimize_threshold(
        self, y_true: np.ndarray, y_probs: np.ndarray, stream_name: str
    ) -> float:
        """
        Finds the probability threshold that maximizes MCC.
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(
            f"[{stream_name}] Optimal Threshold: {best_thresh:.4f}, Max MCC: {best_mcc}"
        )
        return best_thresh

    def train_and_evaluate(
        self, data_train_a: Dict, data_val_a: Dict, data_train_b: Dict, data_val_b: Dict
    ) -> Tuple[Dict[str, xgb.Booster], Dict[str, float]]:
        """
        Main pipeline to train both streams and optimize thresholds.
        """
        models = {}
        thresholds = {}

        # --- Stream A: Interaction ---
        if not data_train_a["X"].empty:
            model_a = self.train_xgboost(
                data_train_a["X"],
                data_train_a["y"],
                data_val_a["X"],
                data_val_a["y"],
                XGB_PARAMS_STREAM_A,
                "stream_a_interaction",
            )
            models["A"] = model_a

            # Predict on validation to find threshold
            dval_a = xgb.DMatrix(data_val_a["X"])
            probs_a = model_a.predict(dval_a)
            thresholds["A"] = self.optimize_threshold(
                data_val_a["y"], probs_a, "Stream A"
            )
        else:
            print("Warning: Stream A training data is empty.")
            models["A"] = None
            thresholds["A"] = 0.5

        # --- Stream B: Impact ---
        if not data_train_b["X"].empty:
            model_b = self.train_xgboost(
                data_train_b["X"],
                data_train_b["y"],
                data_val_b["X"],
                data_val_b["y"],
                XGB_PARAMS_STREAM_B,
                "stream_b_impact",
            )
            models["B"] = model_b

            # Predict on validation to find threshold
            dval_b = xgb.DMatrix(data_val_b["X"])
            probs_b = model_b.predict(dval_b)
            thresholds["B"] = self.optimize_threshold(
                data_val_b["y"], probs_b, "Stream B"
            )
        else:
            print("Warning: Stream B training data is empty.")
            models["B"] = None
            thresholds["B"] = 0.5

        return models, thresholds

    def generate_submission(
        self,
        models: Dict[str, xgb.Booster],
        thresholds: Dict[str, float],
        data_test_a: Dict,
        data_test_b: Dict,
    ):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Generating submission...")

        results = []

        # --- Stream A Predictions ---
        if models.get("A") is not None and not data_test_a["X"].empty:
            dtest_a = xgb.DMatrix(data_test_a["X"])
            probs_a = models["A"].predict(dtest_a)
            preds_a = (probs_a >= thresholds["A"]).astype(int)

            df_res_a = pd.DataFrame(
                {"contact_id": data_test_a["ids"], "contact": preds_a}
            )
            results.append(df_res_a)

        # --- Stream B Predictions ---
        if models.get("B") is not None and not data_test_b["X"].empty:
            dtest_b = xgb.DMatrix(data_test_b["X"])
            probs_b = models["B"].predict(dtest_b)
            preds_b = (probs_b >= thresholds["B"]).astype(int)

            df_res_b = pd.DataFrame(
                {"contact_id": data_test_b["ids"], "contact": preds_b}
            )
            results.append(df_res_b)

        # Combine results
        if results:
            df_submission = pd.concat(results, ignore_index=True)
        else:
            # Fallback if no predictions generated
            print(
                "Warning: No predictions generated. Creating empty submission based on sample."
            )
            df_submission = pd.DataFrame(columns=["contact_id", "contact"])

        # Load sample submission to ensure all IDs are present and order is correct
        if os.path.exists(SAMPLE_SUBMISSION_PATH):
            df_sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

            # Merge predictions into sample
            # We use left join on sample to keep sample order and rows
            df_final = pd.merge(
                df_sample[["contact_id"]], df_submission, on="contact_id", how="left"
            )

            # Fill missing predictions with 0 (no contact)
            df_final["contact"] = df_final["contact"].fillna(0).astype(int)
        else:
            # If sample submission not found (e.g. custom test run), just use what we have
            df_final = df_submission

        # Save
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_final.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}. Shape: {df_final.shape}")
