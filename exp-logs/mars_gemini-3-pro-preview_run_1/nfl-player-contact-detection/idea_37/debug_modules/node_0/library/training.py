import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.data_processing import DataProcessor
from library.models import LGBMWrapper, XGBWrapper


class Trainer:
    def __init__(self):
        self.logger = setup_logger("trainer")
        self.processor = DataProcessor()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        seed_everything(Config.SEED)

    def _get_balanced_subset(self, df, ratio=1.0):
        """
        Returns a balanced subset of the dataframe with all positives and
        a random sample of negatives based on the ratio.
        """
        pos_mask = df["contact"] == 1
        neg_mask = df["contact"] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg_sample = int(n_pos * ratio)

        if n_neg_sample > len(df_neg):
            n_neg_sample = len(df_neg)

        df_neg_sample = df_neg.sample(n=n_neg_sample, random_state=Config.SEED)

        return (
            pd.concat([df_pos, df_neg_sample])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

    def train_scouts(self, df_train, df_val):
        """
        Phase 1: Train Scout models on a balanced dataset.
        """
        self.logger.info("--- Phase 1: Training Dual-Scouts ---")

        # Create balanced scout dataset
        df_scout = self._get_balanced_subset(df_train, ratio=1.0)

        feature_cols = [
            c
            for c in df_train.columns
            if c
            not in [
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
                "contact",
            ]
        ]

        X_train = df_scout[feature_cols]
        y_train = df_scout["contact"]
        X_val = df_val[feature_cols]
        y_val = df_val["contact"]

        # Train Scout LGBM
        scout_lgbm = LGBMWrapper()
        scout_lgbm.train(X_train, y_train, X_val, y_val)
        scout_lgbm.save(os.path.join(self.models_dir, "scout_lgbm.joblib"))

        # Train Scout XGB
        scout_xgb = XGBWrapper()
        scout_xgb.train(X_train, y_train, X_val, y_val)
        scout_xgb.save(os.path.join(self.models_dir, "scout_xgb.joblib"))

        return scout_lgbm, scout_xgb, feature_cols

    def mine_hard_negatives(
        self, df_train, scout_lgbm, scout_xgb, feature_cols, load_cached=True
    ):
        """
        Phase 2: Mine Hard Negatives from the full training set.
        """
        self.logger.info("--- Phase 2: Mining Hard Negatives ---")

        cache_path = os.path.join(Config.WORKING_DIR, "hard_negative_indices.npy")

        if load_cached and os.path.exists(cache_path):
            self.logger.info(f"Loading cached hard negative indices from {cache_path}")
            return np.load(cache_path)

        # Filter for negatives only
        neg_mask = df_train["contact"] == 0
        df_neg = df_train[neg_mask]

        if df_neg.empty:
            self.logger.warning(
                "No negatives found in training set. Returning empty indices."
            )
            return np.array([])

        X_neg = df_neg[feature_cols]

        # Get predictions from both scouts
        preds_lgbm = scout_lgbm.predict(X_neg)
        preds_xgb = scout_xgb.predict(X_neg)

        # Union of hard negatives: Either model thinks prob > threshold
        hard_mask = (preds_lgbm > Config.HARD_NEGATIVE_THRESHOLD) | (
            preds_xgb > Config.HARD_NEGATIVE_THRESHOLD
        )

        hard_indices = df_neg.index[hard_mask].to_numpy()

        self.logger.info(
            f"Mined {len(hard_indices)} hard negatives out of {len(df_neg)} total negatives."
        )

        # Cache results
        np.save(cache_path, hard_indices)

        return hard_indices

    def train_experts(self, df_train, df_val, hard_negative_indices, feature_cols):
        """
        Phase 3: Train Expert models on Positives + Hard Negatives + Anchors.
        """
        self.logger.info("--- Phase 3: Training Anchored Experts ---")

        # 1. Positives
        df_pos = df_train[df_train["contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid (intersection with current df_train index)
        valid_hard_indices = np.intersect1d(df_train.index, hard_negative_indices)
        df_hard = df_train.loc[valid_hard_indices]

        # 3. Anchors (Random Easy Negatives)
        # Negatives that are NOT in hard negatives
        # Note: df_train indices must be unique for this set logic to work perfectly
        all_neg_indices = df_train[df_train["contact"] == 0].index
        easy_indices = np.setdiff1d(all_neg_indices, valid_hard_indices)

        # Sample Anchors: 1:1 ratio with Positives
        n_anchors = int(len(df_pos) * Config.ANCHOR_RATIO)
        if n_anchors > len(easy_indices):
            n_anchors = len(easy_indices)

        rng = np.random.RandomState(Config.SEED)
        anchor_indices = rng.choice(easy_indices, size=n_anchors, replace=False)
        df_anchors = df_train.loc[anchor_indices]

        # Combine
        df_expert = pd.concat([df_pos, df_hard, df_anchors])
        df_expert = df_expert.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        self.logger.info(f"Expert Dataset Size: {len(df_expert)}")
        self.logger.info(
            f"Composition: Pos={len(df_pos)}, HardNeg={len(df_hard)}, Anchors={len(df_anchors)}"
        )

        X_train = df_expert[feature_cols]
        y_train = df_expert["contact"]
        X_val = df_val[feature_cols]
        y_val = df_val["contact"]

        # Train Expert LGBM
        expert_lgbm = LGBMWrapper()
        expert_lgbm.train(X_train, y_train, X_val, y_val)
        expert_lgbm.save(os.path.join(self.models_dir, "expert_lgbm.joblib"))

        # Train Expert XGB
        expert_xgb = XGBWrapper()
        expert_xgb.train(X_train, y_train, X_val, y_val)
        expert_xgb.save(os.path.join(self.models_dir, "expert_xgb.joblib"))

        return expert_lgbm, expert_xgb

    def optimize_threshold(self, expert_lgbm, expert_xgb, df_val, feature_cols):
        """
        Find best threshold on validation set using ensemble predictions.
        """
        self.logger.info("--- Optimization: Threshold Tuning ---")

        X_val = df_val[feature_cols]
        y_val = df_val["contact"].values

        p_lgbm = expert_lgbm.predict(X_val)
        p_xgb = expert_xgb.predict(X_val)

        # Unweighted Ensemble
        p_ens = (p_lgbm * Config.ENSEMBLE_WEIGHTS["lgbm"]) + (
            p_xgb * Config.ENSEMBLE_WEIGHTS["xgb"]
        )

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for t in thresholds:
            preds = (p_ens > t).astype(int)
            mcc = matthews_corrcoef(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        self.logger.info(
            f"Best Validation MCC: {best_mcc:.16f} at Threshold: {best_thresh:.2f}"
        )

        # Save threshold
        np.save(
            os.path.join(self.models_dir, "best_threshold.npy"), np.array([best_thresh])
        )

        return best_thresh

    def run(self, load_cached_data=True):
        """
        Main execution pipeline.
        """
        # 1. Load Data
        df_train = self.processor.get_train_data(
            load_cached=load_cached_data, debug=Config.DEBUG
        )
        df_val = self.processor.get_val_data(
            load_cached=load_cached_data, debug=Config.DEBUG
        )

        # 2. Train Scouts
        scout_lgbm, scout_xgb, feature_cols = self.train_scouts(df_train, df_val)

        # 3. Mine Hard Negatives
        hard_indices = self.mine_hard_negatives(
            df_train, scout_lgbm, scout_xgb, feature_cols, load_cached=load_cached_data
        )

        # 4. Train Experts
        expert_lgbm, expert_xgb = self.train_experts(
            df_train, df_val, hard_indices, feature_cols
        )

        # 5. Optimize
        self.optimize_threshold(expert_lgbm, expert_xgb, df_val, feature_cols)

        self.logger.info("Training pipeline completed successfully.")
