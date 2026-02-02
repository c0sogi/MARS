import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import (
    setup_logging,
    save_joblib,
    load_joblib,
    calc_mcc,
    save_numpy,
    load_numpy,
)
from library.data_manager import DataManager
from library.model_factory import get_estimator, EnsemblePredictor


class Trainer:
    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        self.logger = pd.io.common.logging.getLogger(__name__)
        setup_logging()

        # Ensure model directory exists
        os.makedirs(os.path.join(self.config.WORKING_DIR, "models"), exist_ok=True)

    def _get_model_path(self, name):
        return os.path.join(self.config.WORKING_DIR, "models", f"{name}.joblib")

    def train_scouts(self):
        """
        Phase 1: Train Scout models on a balanced subset of data.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # Load Data
        df_train = self.data_manager.load_train_features()
        df_val = self.data_manager.load_val_features()

        # Prepare Validation Data
        X_val = df_val.drop(columns=["contact"])
        y_val = df_val["contact"]

        # Get Scout Dataset (Balanced)
        X_scout, y_scout = self.data_manager.get_scout_dataset(df_train)

        # Train Scout A: LightGBM
        print("Training Scout A (LightGBM)...")
        scout_lgbm = get_estimator("lgbm")
        scout_lgbm.fit(
            X_scout,
            y_scout,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.TRAINING["EARLY_STOPPING_ROUNDS"],
                    verbose=False,
                )
            ],
        )
        save_joblib(scout_lgbm, self._get_model_path("scout_lgbm"))

        # Train Scout B: XGBoost
        print("Training Scout B (XGBoost)...")
        # Calculate scale_pos_weight for balanced training if needed,
        # but Config says XGB_PARAMS relies on dynamic setting.
        # For Scout, we use a balanced dataset, so scale_pos_weight=1.0 is fine.
        scout_xgb = get_estimator("xgb")
        scout_xgb.fit(X_scout, y_scout, eval_set=[(X_val, y_val)], verbose=False)
        save_joblib(scout_xgb, self._get_model_path("scout_xgb"))

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(self, scouts):
        """
        Phase 2: Use Scouts to mine hard negatives from the full training set.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")

        df_train = self.data_manager.load_train_features()
        X_full = df_train.drop(columns=["contact"])
        y_full = df_train["contact"]

        # Predict with Scouts
        # We only care about negatives
        neg_mask = y_full == 0
        X_neg = X_full[neg_mask]
        indices_neg = X_neg.index

        if len(X_neg) == 0:
            print("No negatives found to mine.")
            return []

        print(f"Scanning {len(X_neg)} negative samples...")

        preds_lgbm = scouts[0].predict_proba(X_neg)[:, 1]
        preds_xgb = scouts[1].predict_proba(X_neg)[:, 1]

        # Hard Negative Condition: Either Scout > Threshold
        threshold = self.config.MINING["SCOUT_THRESHOLD"]
        hard_mask = (preds_lgbm > threshold) | (preds_xgb > threshold)

        hard_indices = indices_neg[hard_mask].to_numpy()

        print(f"Found {len(hard_indices)} hard negatives.")
        self.data_manager.save_hard_negatives(hard_indices)

        return hard_indices

    def train_experts(self, hard_negative_indices):
        """
        Phase 3: Train Expert Ensemble on augmented dataset.
        """
        print("\n--- Phase 3: Training Experts ---")

        df_train = self.data_manager.load_train_features()
        df_val = self.data_manager.load_val_features()

        X_val = df_val.drop(columns=["contact"])
        y_val = df_val["contact"]

        # Construct Expert Dataset
        X_expert, y_expert = self.data_manager.get_expert_dataset(
            df_train, hard_negative_indices
        )

        # 1. Expert LightGBM
        print("Training Expert A (LightGBM)...")
        expert_lgbm = get_estimator("lgbm")
        expert_lgbm.fit(
            X_expert,
            y_expert,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.TRAINING["EARLY_STOPPING_ROUNDS"],
                    verbose=False,
                )
            ],
        )
        save_joblib(expert_lgbm, self._get_model_path("expert_lgbm"))

        # 2. Expert XGBoost
        print("Training Expert B (XGBoost)...")
        # Dynamic scale_pos_weight
        n_neg = (y_expert == 0).sum()
        n_pos = (y_expert == 1).sum()
        scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  XGB scale_pos_weight: {scale_weight:.4f}")

        expert_xgb = get_estimator("xgb")
        expert_xgb.set_params(scale_pos_weight=scale_weight)
        expert_xgb.fit(X_expert, y_expert, eval_set=[(X_val, y_val)], verbose=False)
        save_joblib(expert_xgb, self._get_model_path("expert_xgb"))

        return [expert_lgbm, expert_xgb]

    def optimize_threshold(self, ensemble):
        """
        Phase 4: Optimize decision threshold on Validation set.
        """
        print("\n--- Phase 4: Threshold Optimization ---")

        df_val = self.data_manager.load_val_features()
        X_val = df_val.drop(columns=["contact"])
        y_val = df_val["contact"].values

        # Get probabilities
        print("Generating validation predictions...")
        y_probs = ensemble.predict_proba(X_val)

        # Grid search
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            mcc = calc_mcc(y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Best Threshold: {best_thresh:.2f}")
        print(f"Validation MCC: {best_mcc:.10f}")

        # Save threshold
        save_numpy(
            np.array([best_thresh]),
            os.path.join(self.config.WORKING_DIR, "best_threshold.npy"),
        )
        return best_thresh

    def generate_submission(self, ensemble, threshold):
        """
        Phase 5: Generate Submission for Test Set.
        """
        print("\n--- Phase 5: Generating Submission ---")

        # Load Test Features
        df_test_features = self.data_manager.load_test_features()
        X_test = df_test_features.drop(columns=["contact"], errors="ignore")

        # Predict
        print("Predicting on test set...")
        y_probs = ensemble.predict_proba(X_test)
        y_preds = (y_probs >= threshold).astype(int)

        # Reconstruct Contact IDs
        # FeatureExtractor drops contact_id, so we must reconstruct the order
        # by replicating the split/concat logic on the metadata.
        print("Reconstructing contact_ids...")
        df_meta = pd.read_csv(self.config.TEST_METADATA_PATH)

        # Logic from FeatureExtractor:
        # 1. Split into Ground and Player-Player
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_pg = df_meta[mask_ground].copy()
        df_pp = df_meta[~mask_ground].copy()

        # 2. Concat (PP then PG)
        # Note: FeatureExtractor sorts tracking data but merges on keys.
        # The resulting dataframe order depends on the merge and concat.
        # FeatureExtractor: df_combined = pd.concat([df_pp, df_pg], axis=0)
        # We assume the merge preserves the row order of the left dataframe (metadata)
        # relative to itself, but we must replicate the concat order.

        df_ordered_meta = pd.concat([df_pp, df_pg], axis=0).reset_index(drop=True)

        if len(df_ordered_meta) != len(y_preds):
            print(
                f"WARNING: Metadata rows ({len(df_ordered_meta)}) != Predictions ({len(y_preds)})"
            )
            # Fallback: assume simple read order if lengths mismatch (unlikely given logic)
            # But strictly, the feature extractor logic dictates the order.

        submission = pd.DataFrame(
            {"contact_id": df_ordered_meta["contact_id"], "contact": y_preds}
        )

        # Save
        print(f"Saving submission to {self.config.SUBMISSION_PATH}...")
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print("Submission saved.")

    def run(self):
        """
        Execute the full pipeline.
        """
        print(f"Starting Experiment: {self.config.EXPERIMENT_ID}")

        # 1. Train Scouts
        scout_lgbm, scout_xgb = self.train_scouts()
        scouts = [scout_lgbm, scout_xgb]

        # 2. Mine Hard Negatives
        hard_indices = self.mine_hard_negatives(scouts)

        # 3. Train Experts
        experts = self.train_experts(hard_indices)
        ensemble = EnsemblePredictor(experts)

        # 4. Optimize Threshold
        best_threshold = self.optimize_threshold(ensemble)

        # 5. Generate Submission
        self.generate_submission(ensemble, best_threshold)

        print("Pipeline Completed Successfully.")
