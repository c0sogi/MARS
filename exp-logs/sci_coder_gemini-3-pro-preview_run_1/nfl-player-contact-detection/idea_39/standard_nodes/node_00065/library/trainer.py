import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    HARD_NEGATIVE_INDICES_PATH,
    BEST_THRESHOLD_PATH,
    SCOUT_LGBM_PARAMS,
    SCOUT_XGB_PARAMS,
    LGBM_PARAMS,
    XGB_PARAMS,
    HARD_NEGATIVE_THRESHOLD,
    ANCHOR_RATIO,
    SEED,
)
from library.features import generate_features
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper


class Trainer:
    def __init__(self):
        self.metadata_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
        ]
        self.models_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def _split_X_y(self, df):
        """Separates features and target from the dataframe."""
        # Identify feature columns (all columns not in metadata)
        feature_cols = [c for c in df.columns if c not in self.metadata_cols]

        X = df[feature_cols]
        y = df["contact"] if "contact" in df.columns else None

        return X, y

    def _get_balanced_subset(self, df, ratio=1.0):
        """Creates a balanced subset of Positives and Random Negatives."""
        pos_mask = df["contact"] == 1
        neg_mask = df["contact"] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg_sample = int(n_pos * ratio)

        # Sample negatives
        if len(df_neg) > n_neg_sample:
            df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=SEED)
        else:
            df_neg_sampled = df_neg

        return (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

    def train_scouts(self, df_train):
        """Trains Scout models on a balanced subset."""
        print("--- Phase 1: Training Scouts ---")

        # Prepare Data
        df_scout = self._get_balanced_subset(df_train, ratio=1.0)
        X_scout, y_scout = self._split_X_y(df_scout)

        print(f"Scout Training Data Shape: {X_scout.shape}")

        # Train Scout LGBM
        print("Training Scout LGBM...")
        scout_lgbm = LGBMClassifierWrapper(SCOUT_LGBM_PARAMS)
        scout_lgbm.fit(X_scout, y_scout)
        scout_lgbm.save(os.path.join(self.models_dir, "scout_lgbm.joblib"))

        # Train Scout XGB
        print("Training Scout XGB...")
        scout_xgb = XGBClassifierWrapper(SCOUT_XGB_PARAMS)
        scout_xgb.fit(X_scout, y_scout)
        scout_xgb.save(os.path.join(self.models_dir, "scout_xgb.joblib"))

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(self, df_train, scout_lgbm, scout_xgb, load_cache=True):
        """
        Mines hard negatives using trained scouts.
        Hard Negative: contact=0 AND (P_scout_A > Threshold OR P_scout_B > Threshold)
        """
        print("--- Phase 2: Mining Hard Negatives ---")

        if load_cache and os.path.exists(HARD_NEGATIVE_INDICES_PATH):
            print(f"Loading hard negative indices from {HARD_NEGATIVE_INDICES_PATH}")
            return np.load(HARD_NEGATIVE_INDICES_PATH)

        print("Predicting on full training set with Scouts...")
        X_full, y_full = self._split_X_y(df_train)

        # Get probabilities
        p_lgbm = scout_lgbm.predict_proba(X_full)
        p_xgb = scout_xgb.predict_proba(X_full)

        # Identify Hard Negatives
        # Condition: Is Negative AND (ProbA > Thresh OR ProbB > Thresh)
        is_negative = (y_full == 0).values
        is_hard = (p_lgbm > HARD_NEGATIVE_THRESHOLD) | (p_xgb > HARD_NEGATIVE_THRESHOLD)

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = np.where(hard_negative_mask)[0]

        print(
            f"Mined {len(hard_negative_indices)} Hard Negatives from {np.sum(is_negative)} total negatives."
        )

        # Save to cache
        np.save(HARD_NEGATIVE_INDICES_PATH, hard_negative_indices)

        return hard_negative_indices

    def create_expert_dataset(self, df_train, hard_neg_indices):
        """
        Constructs the Expert Dataset:
        1. All Positives
        2. Mined Hard Negatives
        3. Random Anchors (Easy Negatives) at 1:1 ratio with Positives
        """
        print("Constructing Expert Dataset...")

        # 1. All Positives
        df_pos = df_train[df_train["contact"] == 1].copy()
        n_pos = len(df_pos)

        # 2. Hard Negatives
        df_hard = df_train.iloc[hard_neg_indices].copy()

        # 3. Anchors (Random Negatives)
        # We need to exclude hard negatives from the pool of potential anchors to avoid duplication
        # However, for simplicity and speed (and since hard negs are rare), simple sampling from all negs
        # and dropping duplicates later or ignoring overlap is acceptable.
        # Strictly:
        neg_mask = df_train["contact"] == 0
        # We want indices that are negative but NOT in hard_neg_indices
        # Creating a boolean mask for hard negatives
        hard_mask = np.zeros(len(df_train), dtype=bool)
        hard_mask[hard_neg_indices] = True

        anchor_pool_mask = neg_mask & (~hard_mask)
        df_anchor_pool = df_train[anchor_pool_mask]

        n_anchors = int(n_pos * ANCHOR_RATIO)
        if len(df_anchor_pool) > n_anchors:
            df_anchors = df_anchor_pool.sample(n=n_anchors, random_state=SEED)
        else:
            df_anchors = df_anchor_pool

        print(
            f"Positives: {n_pos}, Hard Negatives: {len(df_hard)}, Anchors: {len(df_anchors)}"
        )

        # Combine
        df_expert = pd.concat([df_pos, df_hard, df_anchors], axis=0)
        df_expert = df_expert.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        return df_expert

    def optimize_threshold(self, y_true, y_prob):
        """Finds the best threshold maximizing MCC."""
        best_mcc = -1.0
        best_thresh = 0.5

        thresholds = np.linspace(0.1, 0.9, 81)  # 0.01 steps
        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def train_experts(self, df_train, hard_neg_indices, df_val):
        """Trains Expert models on the mined dataset."""
        print("--- Phase 3: Training Experts ---")

        # Prepare Train Data
        df_expert = self.create_expert_dataset(df_train, hard_neg_indices)
        X_train, y_train = self._split_X_y(df_expert)

        # Prepare Val Data
        X_val, y_val = self._split_X_y(df_val)

        print(f"Expert Training Data Shape: {X_train.shape}")

        # Train Expert LGBM
        print("Training Expert LGBM...")
        expert_lgbm = LGBMClassifierWrapper(LGBM_PARAMS)
        expert_lgbm.fit(X_train, y_train, X_val, y_val)
        expert_lgbm.save(os.path.join(self.models_dir, "expert_lgbm.joblib"))

        # Train Expert XGB
        print("Training Expert XGB...")
        expert_xgb = XGBClassifierWrapper(XGB_PARAMS)
        expert_xgb.fit(X_train, y_train, X_val, y_val)
        expert_xgb.save(os.path.join(self.models_dir, "expert_xgb.joblib"))

        # Validation & Threshold Optimization
        print("Evaluating Ensemble on Validation Set...")
        p_val_lgbm = expert_lgbm.predict_proba(X_val)
        p_val_xgb = expert_xgb.predict_proba(X_val)
        p_val_ens = (p_val_lgbm + p_val_xgb) / 2.0

        best_thresh, best_mcc = self.optimize_threshold(y_val, p_val_ens)
        print(f"Best Validation MCC: {best_mcc}")
        print(f"Best Threshold: {best_thresh}")

        # Save threshold
        np.save(BEST_THRESHOLD_PATH, np.array([best_thresh]))

        return expert_lgbm, expert_xgb, best_thresh

    def run(self, train_features_cache=True):
        """Executes the full training pipeline."""

        # 1. Load Features
        print("Loading Data...")
        df_train = generate_features(
            split="train", load_cached_data=train_features_cache
        )
        df_val = generate_features(split="val", load_cached_data=train_features_cache)

        # 2. Train Scouts
        scout_lgbm_path = os.path.join(self.models_dir, "scout_lgbm.joblib")
        scout_xgb_path = os.path.join(self.models_dir, "scout_xgb.joblib")

        if os.path.exists(scout_lgbm_path) and os.path.exists(scout_xgb_path):
            print("Loading cached Scouts...")
            scout_lgbm = LGBMClassifierWrapper.load(scout_lgbm_path)
            scout_xgb = XGBClassifierWrapper.load(scout_xgb_path)
        else:
            scout_lgbm, scout_xgb = self.train_scouts(df_train)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(df_train, scout_lgbm, scout_xgb)

        # 4. Train Experts
        expert_lgbm, expert_xgb, best_thresh = self.train_experts(
            df_train, hard_neg_indices, df_val
        )

        # 5. Inference on Test
        print("--- Phase 4: Inference ---")
        df_test = generate_features(split="test", load_cached_data=train_features_cache)
        X_test, _ = self._split_X_y(df_test)

        print("Predicting on Test Set...")
        p_test_lgbm = expert_lgbm.predict_proba(X_test)
        p_test_xgb = expert_xgb.predict_proba(X_test)
        p_test_ens = (p_test_lgbm + p_test_xgb) / 2.0

        # Apply Threshold
        predictions = (p_test_ens > best_thresh).astype(int)

        # Create Submission
        submission = df_test[["contact_id"]].copy()
        submission["contact"] = predictions

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")


def run_training():
    """Entry point wrapper."""
    trainer = Trainer()
    trainer.run()
