import xgboost as xgb
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Tuple, Any

from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    UNDERSAMPLE_RATIO,
    WORKING_DIR,
    SEED,
    SUBMISSION_PATH,
)
from library.utils import compute_mcc


class DualStreamModel:
    """
    Wrapper for the Asymmetric Modality-Selective Dual-Stream GBDT.
    Manages training, threshold optimization, and inference for separate streams.
    """

    def __init__(self):
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.thresholds: Dict[str, float] = {}
        self.streams = ["A", "B"]

    def _undersample(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Performs random undersampling of the majority class (0) to a specific ratio.
        """
        # Ensure reproducibility
        np.random.seed(SEED)

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate target number of negatives
        n_neg_target = int(n_pos * UNDERSAMPLE_RATIO)

        # If we have fewer negatives than target, take all (rare, but safe)
        if n_neg > n_neg_target:
            neg_indices_sampled = np.random.choice(
                neg_indices, size=n_neg_target, replace=False
            )
        else:
            neg_indices_sampled = neg_indices

        # Combine
        indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(indices)

        return X.iloc[indices].copy(), y[indices].copy()

    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Finds the probability threshold that maximizes MCC.
        """
        best_threshold = 0.5
        best_score = -1.0

        # Search space
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        print(
            f"Threshold Optimization - Best MCC: {best_score} at Threshold: {best_threshold}"
        )
        return best_threshold

    def train_stream(
        self,
        stream_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ):
        """
        Trains a single stream's model.
        """
        print(f"\n{'='*20} Training Stream {stream_name} {'='*20}")

        # 1. Undersample Training Data
        print(f"Original Train Shape: {X_train.shape}, Positives: {np.sum(y_train)}")
        X_train_res, y_train_res = self._undersample(X_train, y_train)
        print(
            f"Resampled Train Shape: {X_train_res.shape}, Positives: {np.sum(y_train_res)}"
        )

        # 2. Initialize Model
        # Note: XGB_PARAMS contains 'n_jobs', 'device', 'tree_method', etc.
        model = xgb.XGBClassifier(**XGB_PARAMS)

        # 3. Fit Model
        model.fit(
            X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=VERBOSE_EVAL
        )

        # 4. Optimize Threshold on Validation Set
        # Predict proba returns [prob_0, prob_1], we want prob_1
        val_proba = model.predict_proba(X_val)[:, 1]
        best_thresh = self.optimize_threshold(y_val, val_proba)

        # 5. Save Artifacts
        self.models[stream_name] = model
        self.thresholds[stream_name] = best_thresh

        # Save model to disk
        model_path = os.path.join(WORKING_DIR, f"model_{stream_name}.json")
        model.save_model(model_path)
        print(f"Model for Stream {stream_name} saved to {model_path}")

    def train(self, train_data: Dict[str, Tuple], val_data: Dict[str, Tuple]):
        """
        Orchestrates training for all streams.
        Args:
            train_data: Dict {'A': (X, y, ids), 'B': (X, y, ids)}
            val_data: Dict {'A': (X, y, ids), 'B': (X, y, ids)}
        """
        for stream in self.streams:
            if stream in train_data and stream in val_data:
                X_train, y_train, _ = train_data[stream]
                X_val, y_val, _ = val_data[stream]

                self.train_stream(stream, X_train, y_train, X_val, y_val)
            else:
                print(
                    f"Warning: Data for Stream {stream} missing in train or val sets."
                )

        # Save thresholds
        thresh_path = os.path.join(WORKING_DIR, "thresholds.json")
        with open(thresh_path, "w") as f:
            json.dump(self.thresholds, f, indent=4)
        print(f"Thresholds saved to {thresh_path}")

    def predict(self, test_data: Dict[str, Tuple]) -> pd.DataFrame:
        """
        Generates predictions for test data and creates submission file.
        Args:
            test_data: Dict {'A': (X, y, ids), 'B': (X, y, ids)}
                       Note: y is ignored/placeholder for test.
        """
        results = []

        for stream in self.streams:
            if stream not in test_data:
                continue

            if stream not in self.models:
                print(f"Warning: No trained model found for Stream {stream}. Skipping.")
                continue

            X_test, _, ids_test = test_data[stream]
            model = self.models[stream]
            threshold = self.thresholds.get(stream, 0.5)

            if len(X_test) == 0:
                continue

            # Predict
            proba = model.predict_proba(X_test)[:, 1]
            pred_binary = (proba >= threshold).astype(int)

            # Create DataFrame
            df_stream = pd.DataFrame({"contact_id": ids_test, "contact": pred_binary})
            results.append(df_stream)

        # Combine results
        if results:
            submission_df = pd.concat(results, axis=0)
        else:
            # Fallback if empty (should not happen)
            submission_df = pd.DataFrame(columns=["contact_id", "contact"])

        # Ensure unique contact_ids (safety check)
        submission_df = submission_df.drop_duplicates(subset=["contact_id"])

        # Save Submission
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}. Shape: {submission_df.shape}")

        return submission_df

    def load_models(self):
        """
        Loads trained models and thresholds from disk.
        Useful if skipping training.
        """
        for stream in self.streams:
            model_path = os.path.join(WORKING_DIR, f"model_{stream}.json")
            if os.path.exists(model_path):
                model = xgb.XGBClassifier(**XGB_PARAMS)
                model.load_model(model_path)
                self.models[stream] = model
                print(f"Loaded model for Stream {stream}")

        thresh_path = os.path.join(WORKING_DIR, "thresholds.json")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                self.thresholds = json.load(f)
            print("Loaded thresholds.")
