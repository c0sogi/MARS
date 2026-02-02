import os
import numpy as np
import pandas as pd
import joblib
from library.config import WORKING_DIR, HARD_NEGATIVE_INDICES_PATH, MODEL_DIR, SEED
from library.utils import get_logger, Timer, compute_mcc, seed_everything
from library.data_factory import DataFactory
from library.model_factory import (
    LGBMExpert,
    XGBExpert,
    CatBoostExpert,
    EnsemblePredictor,
)


class Trainer:
    """
    Orchestrates the Tri-Scout Diversity Mining Curriculum.
    """

    def __init__(self):
        self.logger = get_logger("trainer")
        self.data_factory = DataFactory()
        seed_everything(SEED)

        # Threshold file path
        self.threshold_path = os.path.join(WORKING_DIR, "best_threshold.npy")

    def train_scouts(self, df_train):
        """
        Phase 1: Train Tri-Scout models on a balanced dataset.
        """
        self.logger.info("--- Phase 1: Training Scouts ---")

        # 1. Construct Balanced Dataset
        X_scout, y_scout = self.data_factory.get_scout_dataset(df_train, neg_ratio=1.0)

        # 2. Initialize Scouts
        # We use the Expert classes but modify paths to save as scouts
        scout_lgbm = LGBMExpert()
        scout_lgbm.model_path = os.path.join(MODEL_DIR, "scout_lgbm.joblib")

        scout_xgb = XGBExpert()
        scout_xgb.model_path = os.path.join(MODEL_DIR, "scout_xgb.joblib")

        scout_cat = CatBoostExpert()
        scout_cat.model_path = os.path.join(MODEL_DIR, "scout_cat.joblib")

        # 3. Train Scouts
        # Note: Scouts are trained without validation sets to maximize diversity on the balanced subset
        # or we could use a small split, but standard practice for scouts is fit on the sampled set.
        with Timer("Training Scout LGBM", self.logger):
            scout_lgbm.fit(X_scout, y_scout)
            scout_lgbm.save()

        with Timer("Training Scout XGB", self.logger):
            scout_xgb.fit(X_scout, y_scout)
            scout_xgb.save()

        with Timer("Training Scout CatBoost", self.logger):
            scout_cat.fit(X_scout, y_scout)
            scout_cat.save()

        return [scout_lgbm, scout_xgb, scout_cat]

    def mine_hard_negatives(self, scouts, df_train, load_cached_data=True):
        """
        Phase 2: Diversity Mining.
        Identifies hard negatives where ANY scout predicts > 0.05 probability.
        """
        self.logger.info("--- Phase 2: Mining Hard Negatives ---")

        if load_cached_data and os.path.exists(HARD_NEGATIVE_INDICES_PATH):
            self.logger.info(
                f"Loading cached hard negative indices from {HARD_NEGATIVE_INDICES_PATH}"
            )
            return np.load(HARD_NEGATIVE_INDICES_PATH)

        self.logger.info("Running mining process on full training set...")

        # We need to predict on the full df_train (gated survivors)
        # Extract features for prediction
        X_full = self.data_factory.get_test_data(
            df_train
        )  # get_test_data just returns X

        # Get probabilities from each scout
        probs_lgbm = scouts[0].predict(X_full)
        probs_xgb = scouts[1].predict(X_full)
        probs_cat = scouts[2].predict(X_full)

        # Mining Logic: Union of errors
        # Hard Negative if: Actual Contact == 0 AND (Prob_A > 0.05 OR Prob_B > 0.05 OR Prob_C > 0.05)
        mining_threshold = 0.05

        is_negative = (df_train["contact"] == 0).values
        is_hard = (
            (probs_lgbm > mining_threshold)
            | (probs_xgb > mining_threshold)
            | (probs_cat > mining_threshold)
        )

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = np.where(hard_negative_mask)[0]

        self.logger.info(
            f"Mined {len(hard_negative_indices)} hard negatives out of {np.sum(is_negative)} total negatives."
        )

        # Save to cache
        np.save(HARD_NEGATIVE_INDICES_PATH, hard_negative_indices)
        self.logger.info(f"Saved hard negative indices to {HARD_NEGATIVE_INDICES_PATH}")

        return hard_negative_indices

    def train_experts(self, df_train, hard_neg_indices, df_val):
        """
        Phase 3: Train Expert Ensemble on enriched dataset.
        """
        self.logger.info("--- Phase 3: Training Experts ---")

        # 1. Construct Expert Dataset
        X_train, y_train = self.data_factory.get_expert_dataset(
            df_train, hard_neg_indices, buffer_ratio=0.1
        )

        # 2. Prepare Validation Data
        X_val, y_val = self.data_factory.get_validation_data(df_val)

        # 3. Initialize Experts (Default paths)
        expert_lgbm = LGBMExpert()
        expert_xgb = XGBExpert()
        expert_cat = CatBoostExpert()

        # 4. Train Experts
        with Timer("Training Expert LGBM", self.logger):
            expert_lgbm.fit(X_train, y_train, X_val, y_val)
            expert_lgbm.save()

        with Timer("Training Expert XGB", self.logger):
            expert_xgb.fit(X_train, y_train, X_val, y_val)
            expert_xgb.save()

        with Timer("Training Expert CatBoost", self.logger):
            expert_cat.fit(X_train, y_train, X_val, y_val)
            expert_cat.save()

        return [expert_lgbm, expert_xgb, expert_cat]

    def optimize_threshold(self, ensemble, df_val):
        """
        Finds the decision threshold that maximizes MCC on the validation set.
        """
        self.logger.info("--- Optimizing Threshold ---")

        X_val, y_val = self.data_factory.get_validation_data(df_val)

        # Get ensemble probability
        y_prob = ensemble.predict(X_val)

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            mcc = compute_mcc(y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        self.logger.info(f"Best Validation MCC: {best_mcc}")  # Full precision
        self.logger.info(f"Best Threshold: {best_thresh}")

        # Save threshold
        np.save(self.threshold_path, np.array([best_thresh]))

        return best_thresh

    def run_pipeline(self, load_cached_features=True, load_cached_mining=True):
        """
        Executes the full training pipeline.
        """
        with Timer("Full Training Pipeline", self.logger):
            # 1. Load Data
            df_train = self.data_factory.load_features(
                mode="train", load_cached_data=load_cached_features
            )
            df_val = self.data_factory.load_features(
                mode="val", load_cached_data=load_cached_features
            )

            # 2. Train Scouts
            scouts = self.train_scouts(df_train)

            # 3. Mine Hard Negatives
            hard_neg_indices = self.mine_hard_negatives(
                scouts, df_train, load_cached_data=load_cached_mining
            )

            # 4. Train Experts
            experts = self.train_experts(df_train, hard_neg_indices, df_val)
            ensemble = EnsemblePredictor(experts)

            # 5. Optimize Threshold
            best_thresh = self.optimize_threshold(ensemble, df_val)

            return ensemble, best_thresh

    def load_best_ensemble(self):
        """
        Loads trained expert models and the optimized threshold.
        """
        expert_lgbm = LGBMExpert()
        expert_xgb = XGBExpert()
        expert_cat = CatBoostExpert()

        if not (expert_lgbm.load() and expert_xgb.load() and expert_cat.load()):
            raise FileNotFoundError("Could not load one or more expert models.")

        ensemble = EnsemblePredictor([expert_lgbm, expert_xgb, expert_cat])

        if os.path.exists(self.threshold_path):
            best_thresh = float(np.load(self.threshold_path)[0])
        else:
            self.logger.warning("Threshold file not found, defaulting to 0.5")
            best_thresh = 0.5

        return ensemble, best_thresh
