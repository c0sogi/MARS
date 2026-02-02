import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
import joblib
from library.config import Config
from library.utils import calc_mcc


class DualStreamModel:
    """
    Implements the Physically-Disentangled Dual-Stream GBDT.
    Manages two separate XGBoost models:
    - Stream A: Interaction Model (Player-Player)
    - Stream B: Impact Model (Player-Ground)
    """

    def __init__(self):
        self.config = Config
        self.models = {}
        self.thresholds = {}
        self.features = {}

        # Columns to exclude from features
        self.ignore_cols = [
            "contact_id",
            "game_play",
            "game_key",
            "play_id",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
            "frame_approx",
            "time",
            "jersey_number",  # Treat as metadata/noise
            "team",  # Treat as metadata for now, or could be categorical
            "position",  # Treat as metadata
        ]

    def _get_feature_cols(self, df):
        """
        Identifies feature columns by excluding metadata and targets.
        """
        return [c for c in df.columns if c not in self.ignore_cols]

    def _undersample_negatives(self, df):
        """
        Applies Targeted Majority Undersampling.
        Retains 100% of positives and samples negatives to achieve the configured ratio.
        """
        positives = df[df["contact"] == 1]
        negatives = df[df["contact"] == 0]

        n_pos = len(positives)
        if n_pos == 0:
            print("Warning: No positive samples found in training data.")
            return df

        ratio = self.config.TRAIN_CONFIG["neg_pos_ratio"]
        n_neg = int(n_pos * ratio)

        # Ensure we don't sample more negatives than exist
        n_neg = min(n_neg, len(negatives))

        print(
            f"Undersampling: Positives={n_pos}, Negatives={len(negatives)} -> {n_neg}"
        )

        negatives_sampled = negatives.sample(n=n_neg, random_state=self.config.SEED)

        df_sampled = pd.concat([positives, negatives_sampled], axis=0)
        # Shuffle
        df_sampled = df_sampled.sample(
            frac=1, random_state=self.config.SEED
        ).reset_index(drop=True)

        return df_sampled

    def _optimize_threshold(self, y_true, y_prob):
        """
        Finds the probability threshold that maximizes MCC.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Search space: 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            score = calc_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        return best_threshold, best_mcc

    def _train_stream(self, df_train, df_val, stream_name):
        """
        Trains a single stream (A or B).
        """
        print(f"\n[{stream_name}] Starting training pipeline...")

        # 1. Feature Selection
        feature_cols = self._get_feature_cols(df_train)
        self.features[stream_name] = feature_cols
        print(f"[{stream_name}] Selected {len(feature_cols)} features.")

        # 2. Undersampling (Train only)
        df_train_sampled = self._undersample_negatives(df_train)

        # 3. Prepare DMatrix
        X_train = df_train_sampled[feature_cols]
        y_train = df_train_sampled["contact"]

        X_val = df_val[feature_cols]
        y_val = df_val["contact"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 4. Configuration
        if stream_name == "stream_a":
            params = self.config.XGB_PARAMS_STREAM_A.copy()
        else:
            params = self.config.XGB_PARAMS_STREAM_B.copy()

        # 5. Training
        print(f"[{stream_name}] Training XGBoost...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.TRAIN_CONFIG["num_boost_round"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.config.TRAIN_CONFIG["early_stopping_rounds"],
            verbose_eval=self.config.TRAIN_CONFIG["verbose_eval"],
        )

        self.models[stream_name] = model

        # 6. Threshold Optimization
        if self.config.TRAIN_CONFIG["optimize_threshold"]:
            print(f"[{stream_name}] Optimizing threshold on validation set...")
            y_prob_val = model.predict(dval)
            best_thresh, best_mcc = self._optimize_threshold(y_val, y_prob_val)
            self.thresholds[stream_name] = best_thresh
            print(
                f"[{stream_name}] Best Threshold: {best_thresh:.4f}, Validation MCC: {best_mcc:.16f}"
            )
        else:
            self.thresholds[stream_name] = 0.5
            print(f"[{stream_name}] Using default threshold: 0.5")

        # Cleanup
        del dtrain, dval, X_train, X_val, df_train_sampled
        gc.collect()

    def train(self, train_a, train_b, val_a, val_b):
        """
        Main training entry point. Trains both streams.

        Args:
            train_a, train_b: Training DataFrames for Stream A and B.
            val_a, val_b: Validation DataFrames for Stream A and B.
        """
        # Train Stream A (Interaction)
        if not train_a.empty and not val_a.empty:
            self._train_stream(train_a, val_a, "stream_a")
        else:
            print("Warning: Stream A data empty. Skipping Stream A training.")

        # Train Stream B (Impact)
        if not train_b.empty and not val_b.empty:
            self._train_stream(train_b, val_b, "stream_b")
        else:
            print("Warning: Stream B data empty. Skipping Stream B training.")

    def predict(self, test_a, test_b):
        """
        Generates predictions for the test set.

        Args:
            test_a: Test DataFrame for Stream A.
            test_b: Test DataFrame for Stream B.

        Returns:
            pd.DataFrame: Combined predictions with columns ['contact_id', 'contact']
        """
        results = []

        # Predict Stream A
        if not test_a.empty and "stream_a" in self.models:
            print("[Predict] Generating predictions for Stream A...")
            features_a = self.features["stream_a"]
            dtest_a = xgb.DMatrix(test_a[features_a])
            probs_a = self.models["stream_a"].predict(dtest_a)
            preds_a = (probs_a >= self.thresholds["stream_a"]).astype(int)

            res_a = pd.DataFrame(
                {"contact_id": test_a["contact_id"], "contact": preds_a}
            )
            results.append(res_a)
            del dtest_a
            gc.collect()
        elif not test_a.empty:
            # Fallback if model missing but data exists (shouldn't happen in valid run)
            print("[Predict] Warning: Stream A model missing. Predicting 0.")
            res_a = pd.DataFrame({"contact_id": test_a["contact_id"], "contact": 0})
            results.append(res_a)

        # Predict Stream B
        if not test_b.empty and "stream_b" in self.models:
            print("[Predict] Generating predictions for Stream B...")
            features_b = self.features["stream_b"]
            dtest_b = xgb.DMatrix(test_b[features_b])
            probs_b = self.models["stream_b"].predict(dtest_b)
            preds_b = (probs_b >= self.thresholds["stream_b"]).astype(int)

            res_b = pd.DataFrame(
                {"contact_id": test_b["contact_id"], "contact": preds_b}
            )
            results.append(res_b)
            del dtest_b
            gc.collect()
        elif not test_b.empty:
            print("[Predict] Warning: Stream B model missing. Predicting 0.")
            res_b = pd.DataFrame({"contact_id": test_b["contact_id"], "contact": 0})
            results.append(res_b)

        # Combine
        if results:
            df_submission = pd.concat(results, axis=0)
        else:
            df_submission = pd.DataFrame(columns=["contact_id", "contact"])

        return df_submission

    def save_submission(self, df_submission):
        """
        Saves the submission file to the configured path.
        """
        output_path = self.config.PATH_CONFIG["submission_path"]
        print(f"Saving submission to {output_path}...")

        # Ensure we match the sample submission format (if any rows are missing, fill 0)
        # Load sample submission to get exhaustive list of contact_ids
        sample_sub = pd.read_csv(self.config.PATH_CONFIG["sample_submission"])

        # Merge predictions onto sample submission to ensure order and completeness
        final_sub = pd.merge(
            sample_sub[["contact_id"]], df_submission, on="contact_id", how="left"
        )

        # Fill missing with 0 (no contact)
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        final_sub.to_csv(output_path, index=False)
        print(f"Submission saved. Shape: {final_sub.shape}")
