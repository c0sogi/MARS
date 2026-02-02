import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, matthews_corrcoef
from library.config import Config
from library.utils import (
    setup_logger,
    save_joblib,
    load_joblib,
    save_npy,
    load_npy,
    compute_mcc,
)
from library.feature_engine import FeatureEngine
from library.model_zoo import LGBMWrapper, XGBWrapper, EnsemblePredictor


class TrainingPipeline:
    """
    Orchestrates the Dual-Scout Anchored Mining training curriculum.
    """

    def __init__(self):
        self.logger = setup_logger(name="training_pipeline")
        self.feature_cols = None  # To be populated after loading data

    def _get_feature_columns(self, df):
        """
        Identifies feature columns (excluding metadata).
        """
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
            "step_offset",
        ]
        return [c for c in df.columns if c not in exclude_cols]

    def _balance_dataset(self, df):
        """
        Creates a balanced dataset (1:1 Positive:Negative) for Scout training.
        """
        pos_mask = df["contact"] == 1
        neg_mask = df["contact"] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        # Sample negatives to match positives
        n_pos = len(df_pos)
        if n_pos == 0:
            raise ValueError("No positive samples found in training data.")

        # If not enough negatives, take all
        n_neg_sample = min(len(df_neg), n_pos)

        df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=Config.SEED)

        df_balanced = (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )
        return df_balanced

    def _train_scouts(self, df_train, df_val, load_cached_models=True):
        """
        Phase 1: Train Scout Models on balanced data.
        """
        scout_lgbm_path = "models/scout_lgbm.joblib"
        scout_xgb_path = "models/scout_xgb.joblib"

        # Check cache
        if load_cached_models:
            scout_lgbm = LGBMWrapper.load(scout_lgbm_path)
            scout_xgb = XGBWrapper.load(scout_xgb_path)
            if scout_lgbm is not None and scout_xgb is not None:
                self.logger.info("Loaded Scout models from cache.")
                return scout_lgbm, scout_xgb

        self.logger.info("Training Scout models...")

        # Prepare Balanced Data
        df_balanced = self._balance_dataset(df_train)

        X_train = df_balanced[self.feature_cols]
        y_train = df_balanced["contact"]

        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"]

        # Train LGBM Scout
        self.logger.info("Training Scout LGBM...")
        scout_lgbm = LGBMWrapper()
        scout_lgbm.fit(X_train, y_train, X_val, y_val)
        scout_lgbm.save(scout_lgbm_path)

        # Train XGB Scout
        self.logger.info("Training Scout XGB...")
        scout_xgb = XGBWrapper()
        scout_xgb.fit(X_train, y_train, X_val, y_val)
        scout_xgb.save(scout_xgb_path)

        return scout_lgbm, scout_xgb

    def _mine_hard_negatives(
        self, df_train, scout_lgbm, scout_xgb, load_cached_mining=True
    ):
        """
        Phase 2: Mine Hard Negatives using Scouts.
        Hard Negatives are negative samples (y=0) where P(contact) > Threshold.
        """
        mining_cache_path = "hard_negative_indices.npy"

        if load_cached_mining:
            indices = load_npy(mining_cache_path)
            if indices is not None:
                self.logger.info(
                    f"Loaded {len(indices)} hard negative indices from cache."
                )
                return indices

        self.logger.info("Mining Hard Negatives...")

        # Filter for actual negatives first to avoid predicting on positives (waste of compute)
        # Note: We need the original indices relative to df_train to extract them later.
        # So we predict on everything or carefully map indices.
        # Predicting on everything is safer for index alignment.

        X_full = df_train[self.feature_cols]

        # Get probabilities
        p_lgbm = scout_lgbm.predict_proba(X_full)
        p_xgb = scout_xgb.predict_proba(X_full)

        # Identify Hard Negatives
        # Condition: Ground Truth is 0 AND (ScoutA > Thresh OR ScoutB > Thresh)
        is_negative = (df_train["contact"] == 0).values
        is_hard = (p_lgbm > Config.HARD_NEGATIVE_THRESHOLD) | (
            p_xgb > Config.HARD_NEGATIVE_THRESHOLD
        )

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = np.where(hard_negative_mask)[0]

        self.logger.info(
            f"Found {len(hard_negative_indices)} hard negatives out of {np.sum(is_negative)} total negatives."
        )

        save_npy(hard_negative_indices, mining_cache_path)

        return hard_negative_indices

    def _construct_expert_dataset(self, df_train, hard_neg_indices):
        """
        Constructs the dataset for Expert training:
        1. All Positives
        2. Mined Hard Negatives
        3. Random Anchor Negatives (1:1 ratio with Positives)
        """
        # 1. All Positives
        df_pos = df_train[df_train["contact"] == 1].copy()
        n_pos = len(df_pos)

        # 2. Hard Negatives
        df_hard = df_train.iloc[hard_neg_indices].copy()

        # 3. Random Anchors
        # Exclude hard negatives and positives from the pool
        # We can do this by index exclusion
        all_indices = df_train.index.values
        pos_indices = df_pos.index.values

        exclude_indices = np.union1d(pos_indices, hard_neg_indices)
        candidate_anchor_indices = np.setdiff1d(all_indices, exclude_indices)

        # Sample Anchors
        n_anchors = int(n_pos * Config.ANCHOR_RATIO)
        # Ensure we don't sample more than available
        n_anchors = min(n_anchors, len(candidate_anchor_indices))

        rng = np.random.RandomState(Config.SEED)
        anchor_indices = rng.choice(
            candidate_anchor_indices, size=n_anchors, replace=False
        )
        df_anchors = df_train.loc[anchor_indices].copy()

        # Combine
        df_expert = pd.concat([df_pos, df_hard, df_anchors], axis=0)
        df_expert = df_expert.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        self.logger.info(
            f"Expert Dataset Stats: Pos={len(df_pos)}, HardNeg={len(df_hard)}, Anchors={len(df_anchors)}, Total={len(df_expert)}"
        )

        return df_expert

    def _train_experts(
        self, df_train, df_val, hard_neg_indices, load_cached_models=True
    ):
        """
        Phase 3: Train Expert Models on the constructed dataset.
        """
        expert_lgbm_path = "models/expert_lgbm.joblib"
        expert_xgb_path = "models/expert_xgb.joblib"

        if load_cached_models:
            expert_lgbm = LGBMWrapper.load(expert_lgbm_path)
            expert_xgb = XGBWrapper.load(expert_xgb_path)
            if expert_lgbm is not None and expert_xgb is not None:
                self.logger.info("Loaded Expert models from cache.")
                return expert_lgbm, expert_xgb

        self.logger.info("Constructing Expert Dataset...")
        df_expert = self._construct_expert_dataset(df_train, hard_neg_indices)

        X_train = df_expert[self.feature_cols]
        y_train = df_expert["contact"]

        # Validation is always on the full validation set (or gated validation set)
        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"]

        # Train LGBM Expert
        self.logger.info("Training Expert LGBM...")
        expert_lgbm = LGBMWrapper()
        expert_lgbm.fit(X_train, y_train, X_val, y_val)
        expert_lgbm.save(expert_lgbm_path)

        # Train XGB Expert
        self.logger.info("Training Expert XGB...")
        expert_xgb = XGBWrapper()
        expert_xgb.fit(X_train, y_train, X_val, y_val)
        expert_xgb.save(expert_xgb_path)

        return expert_lgbm, expert_xgb

    def _optimize_threshold(self, y_true, y_prob):
        """
        Finds the best threshold maximizing MCC.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        thresholds = np.arange(0.01, 1.00, 0.01)

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        return best_threshold, best_mcc

    def run(self, load_cached_data=True, sample_size=None):
        """
        Main execution method for the pipeline.
        """
        self.logger.info("Starting Training Pipeline...")

        # 1. Feature Generation
        df_train = FeatureEngine.generate_features(
            split="train", load_cached_data=load_cached_data, sample_size=sample_size
        )
        df_val = FeatureEngine.generate_features(
            split="val", load_cached_data=load_cached_data, sample_size=sample_size
        )

        if df_train.empty or df_val.empty:
            self.logger.error("Feature generation returned empty dataframe.")
            return

        self.feature_cols = self._get_feature_columns(df_train)
        self.logger.info(f"Feature columns identified: {len(self.feature_cols)}")

        # 2. Phase 1: Train Scouts
        scout_lgbm, scout_xgb = self._train_scouts(
            df_train, df_val, load_cached_models=load_cached_data
        )

        # 3. Phase 2: Mine Hard Negatives
        hard_neg_indices = self._mine_hard_negatives(
            df_train, scout_lgbm, scout_xgb, load_cached_mining=load_cached_data
        )

        # 4. Phase 3: Train Experts
        expert_lgbm, expert_xgb = self._train_experts(
            df_train, df_val, hard_neg_indices, load_cached_models=load_cached_data
        )

        # 5. Validation & Threshold Optimization
        self.logger.info("Evaluating Expert Ensemble on Validation Set...")

        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"].values

        # Ensemble Prediction
        p_lgbm = expert_lgbm.predict_proba(X_val)
        p_xgb = expert_xgb.predict_proba(X_val)
        p_ensemble = (p_lgbm + p_xgb) / 2.0

        val_log_loss = log_loss(y_val, p_ensemble)
        self.logger.info(f"Validation LogLoss (Ensemble): {val_log_loss}")

        # Optimize Threshold
        best_thresh, best_mcc = self._optimize_threshold(y_val, p_ensemble)
        self.logger.info(f"Best Threshold: {best_thresh}")
        self.logger.info(f"Best Validation MCC: {best_mcc}")

        # Save Threshold
        save_npy(np.array([best_thresh]), "models/best_threshold.npy")

        # Cleanup
        del df_train, df_val, X_val, p_lgbm, p_xgb, p_ensemble
        gc.collect()

        self.logger.info("Training Pipeline Completed Successfully.")
