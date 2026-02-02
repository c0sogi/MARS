import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    setup_logger,
    save_model,
    load_model,
    save_cache_npy,
    load_cache_npy,
    compute_mcc,
)
from library.data_manager import DataManager
from library.model_zoo import LGBMExpert, XGBExpert, CatBoostExpert, EnsemblePredictor


class Trainer:
    """
    Orchestrates the Dual-Scout Diversity Mining training curriculum.
    """

    def __init__(self):
        self.logger = setup_logger("trainer")
        self.data_manager = DataManager()

    def train_scouts(self, df_train, X_val, y_val):
        """
        Phase 1: Train Scout models (LGBM, XGB) on a balanced dataset.
        """
        self.logger.info("--- Phase 1: Training Scouts ---")

        # 1. Get Balanced Dataset
        X_scout, y_scout = self.data_manager.get_scout_dataset(df_train)

        # 2. Train LightGBM Scout
        self.logger.info("Training Scout A (LightGBM)...")
        # We use slightly reduced epochs for scouts if desired, but config controls it
        # Using a copy of params to avoid modifying global config if we wanted to tweak
        scout_lgbm = LGBMExpert()
        scout_lgbm.fit(X_scout, y_scout, X_val, y_val)
        save_model(scout_lgbm, "scout_lgbm.joblib")

        # 3. Train XGBoost Scout
        self.logger.info("Training Scout B (XGBoost)...")
        scout_xgb = XGBExpert()
        scout_xgb.fit(X_scout, y_scout, X_val, y_val)
        save_model(scout_xgb, "scout_xgb.joblib")

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(self, df_train, scout_lgbm, scout_xgb, load_cached=True):
        """
        Phase 2: Use Scouts to mine hard negatives from the full training pool.
        Caches the indices of hard negatives.
        """
        self.logger.info("--- Phase 2: Mining Hard Negatives ---")
        cache_filename = "hard_negative_indices.npy"

        # Check Cache
        if load_cached:
            cached_indices = load_cache_npy(cache_filename)
            if cached_indices is not None:
                self.logger.info(
                    f"Loaded {len(cached_indices)} hard negative indices from cache."
                )
                return cached_indices

        self.logger.info("Mining hard negatives from scratch...")

        # Get Feature Matrix for full training set
        X_full = self.data_manager.get_feature_matrix(df_train)

        # Get Predictions
        # We only care about negatives, but predicting on all is usually vectorized and fast enough
        # Optimization: Filter for negatives first to save inference time if dataset is huge
        neg_mask = df_train["contact"] == 0
        neg_indices = df_train[neg_mask].index.values
        X_neg = X_full[neg_mask]

        if len(X_neg) == 0:
            self.logger.warning(
                "No negatives found in training set. Returning empty list."
            )
            return np.array([])

        prob_lgbm = scout_lgbm.predict_proba(X_neg)
        prob_xgb = scout_xgb.predict_proba(X_neg)

        # Union Logic: Hard if P(Contact) > Threshold in EITHER model
        threshold = Config.TRAINING["SCOUT_MINING_THRESHOLD"]
        is_hard = (prob_lgbm > threshold) | (prob_xgb > threshold)

        hard_indices = neg_indices[is_hard]

        self.logger.info(
            f"Mined {len(hard_indices)} hard negatives out of {len(neg_indices)} total negatives."
        )

        # Cache results
        save_cache_npy(hard_indices, cache_filename)

        return hard_indices

    def train_expert_ensemble(self, df_train, hard_neg_indices, X_val, y_val):
        """
        Phase 3: Train the Tri-Model Expert Ensemble on the enriched dataset.
        """
        self.logger.info("--- Phase 3: Training Expert Ensemble ---")

        # 1. Construct Expert Dataset
        X_expert, y_expert = self.data_manager.get_expert_dataset(
            df_train, hard_neg_indices
        )

        # 2. Train LightGBM Expert
        self.logger.info("Training Expert A (LightGBM)...")
        expert_lgbm = LGBMExpert()
        expert_lgbm.fit(X_expert, y_expert, X_val, y_val)
        save_model(expert_lgbm, "expert_lgbm.joblib")

        # 3. Train XGBoost Expert
        self.logger.info("Training Expert B (XGBoost)...")
        expert_xgb = XGBExpert()
        expert_xgb.fit(X_expert, y_expert, X_val, y_val)
        save_model(expert_xgb, "expert_xgb.joblib")

        # 4. Train CatBoost Expert
        self.logger.info("Training Expert C (CatBoost)...")
        expert_cat = CatBoostExpert()
        expert_cat.fit(X_expert, y_expert, X_val, y_val)
        # CatBoost model saving is handled slightly differently usually, but joblib works for the wrapper
        save_model(expert_cat, "expert_cat.joblib")

        # Create Ensemble
        ensemble = EnsemblePredictor(models=[expert_lgbm, expert_xgb, expert_cat])
        return ensemble

    def optimize_threshold(self, ensemble, X_val, y_val):
        """
        Finds the probability threshold that maximizes MCC on the validation set.
        """
        self.logger.info("--- Threshold Optimization ---")

        probs = ensemble.predict(X_val)

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = compute_mcc(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        self.logger.info(f"Best Validation MCC: {best_mcc}")
        self.logger.info(f"Best Threshold: {best_thresh}")

        # Save threshold
        save_cache_npy(np.array([best_thresh]), "best_threshold.npy")

        return best_thresh

    def generate_submission(self, ensemble, threshold, df_test):
        """
        Generates predictions for the test set and saves the submission file.
        """
        self.logger.info("--- Generating Submission ---")

        X_test, contact_ids = self.data_manager.get_test_set(df_test)

        if len(X_test) == 0:
            self.logger.warning("Test set is empty!")
            return

        probs = ensemble.predict(X_test)
        preds = (probs >= threshold).astype(int)

        submission = pd.DataFrame({"contact_id": contact_ids, "contact": preds})

        # Ensure output directory exists (handled by Config, but safe to check)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(
            f"Submission saved to {Config.SUBMISSION_PATH} with {len(submission)} rows."
        )

    def run_pipeline(self, load_cached_features=True, load_cached_mining=True):
        """
        Main execution entry point.
        """
        self.logger.info("Starting QGSM-E Training Pipeline...")

        # 1. Load Data
        df_train = self.data_manager.load_train_features(
            load_cached=load_cached_features
        )
        df_val = self.data_manager.load_val_features(load_cached=load_cached_features)
        df_test = self.data_manager.load_test_features(load_cached=load_cached_features)

        X_val, y_val = self.data_manager.get_validation_set(df_val)

        # 2. Phase 1: Scouts
        # Check if scouts exist to skip training? For now, we train fresh or rely on internal logic if needed.
        # Assuming we train fresh every run or rely on upstream caching.
        # Here we train fresh for the session.
        scout_lgbm, scout_xgb = self.train_scouts(df_train, X_val, y_val)

        # Clean up memory
        gc.collect()

        # 3. Phase 2: Mining
        hard_neg_indices = self.mine_hard_negatives(
            df_train, scout_lgbm, scout_xgb, load_cached=load_cached_mining
        )

        # Clean up scouts if memory is tight, but we might keep them.
        del scout_lgbm, scout_xgb
        gc.collect()

        # 4. Phase 3: Experts
        ensemble = self.train_expert_ensemble(df_train, hard_neg_indices, X_val, y_val)

        # 5. Optimization
        best_threshold = self.optimize_threshold(ensemble, X_val, y_val)

        # 6. Submission
        self.generate_submission(ensemble, best_threshold, df_test)

        self.logger.info("Pipeline completed successfully.")
