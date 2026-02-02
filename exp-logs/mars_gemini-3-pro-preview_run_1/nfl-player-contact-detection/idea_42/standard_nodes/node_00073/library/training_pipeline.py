import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from library.config import WORKING_DIR, SEED, ANCHOR_RATIO
from library.utils import seed_everything, calc_mcc, optimize_threshold, cache_result
from library.data_loader import DataLoader
from library.model_factory import get_model, save_model, EnsemblePredictor

# Define non-feature columns to exclude from training
NON_FEATURE_COLS = [
    "contact_id",
    "game_play",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "step",
    "datetime",
    "contact",
    "video_path_endzone",
    "video_path_sideline",
    "video_path_all29",
    "actual_step",  # Intermediate col if leaked
    "join_id_2",  # Intermediate col if leaked
]


class TrainingPipeline:
    """
    Orchestrates the Kinematically-Aligned Momentum-Anchored Ensemble (KAM-AE) training pipeline.
    """

    @staticmethod
    def _get_feature_cols(df):
        """Helper to identify feature columns."""
        return [c for c in df.columns if c not in NON_FEATURE_COLS]

    @staticmethod
    def train_scouts(train_df, val_df):
        """
        Phase 1: Train Scout models (LGBM & XGB) on a balanced subset of gated survivors.
        """
        print("\n[TrainingPipeline] Phase 1: Training Dual-Scout Models...")

        # 1. Create Balanced Dataset
        scout_data = DataLoader.sample_balanced_scout_data(train_df, random_state=SEED)

        feature_cols = TrainingPipeline._get_feature_cols(scout_data)
        X_scout = scout_data[feature_cols]
        y_scout = scout_data["contact"]

        # Validation set for early stopping
        X_val = val_df[feature_cols]
        y_val = val_df["contact"]

        models = {}

        # 2. Train LightGBM Scout
        print(" -> Training Scout A (LightGBM)...")
        lgbm_scout = get_model("lgbm", random_state=SEED)
        lgbm_scout.fit(
            X_scout,
            y_scout,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            callbacks=[
                # LightGBM python-package callbacks
                # Note: early_stopping is handled via fit params in newer sklearn API wrappers usually,
                # but explicit callback is safer for some versions.
                # However, sklearn API supports early_stopping_rounds directly in fit.
            ],
        )
        models["lgbm"] = lgbm_scout
        save_model(lgbm_scout, os.path.join(WORKING_DIR, "models/scout_lgbm.joblib"))

        # 3. Train XGBoost Scout
        print(" -> Training Scout B (XGBoost)...")
        xgb_scout = get_model("xgb", random_state=SEED)
        xgb_scout.fit(X_scout, y_scout, eval_set=[(X_val, y_val)], verbose=False)
        models["xgb"] = xgb_scout
        save_model(xgb_scout, os.path.join(WORKING_DIR, "models/scout_xgb.joblib"))

        return models

    @staticmethod
    def mine_hard_negatives(scout_models, train_df, load_cached_data=True):
        """
        Phase 2: Mine Hard Negatives from the full gated training set.
        Hard Negatives are negative samples where P(Contact) > 0.05 by EITHER scout.

        Implements strict caching logic.
        """
        print("\n[TrainingPipeline] Phase 2: Mining Hard Negatives...")

        filename = "hard_negative_indices.npy"
        filepath = os.path.join(WORKING_DIR, filename)

        # 1. Try Load
        if load_cached_data and os.path.exists(filepath):
            print(f"[Cache] Loading hard negative indices from {filepath}...")
            return np.load(filepath, allow_pickle=True)

        # 2. Compute
        print(" -> Computing hard negatives (Running Scouts on full train set)...")
        feature_cols = TrainingPipeline._get_feature_cols(train_df)
        X_full = train_df[feature_cols]

        # Get probabilities
        p_lgbm = scout_models["lgbm"].predict_proba(X_full)[:, 1]
        p_xgb = scout_models["xgb"].predict_proba(X_full)[:, 1]

        # Identify Hard Negatives
        # Condition: (True Label == 0) AND ((Prob_LGBM > 0.05) OR (Prob_XGB > 0.05))
        is_negative = (train_df["contact"] == 0).values
        is_hard = (p_lgbm > 0.05) | (p_xgb > 0.05)

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = train_df.index[hard_negative_mask].to_numpy()

        print(
            f" -> Found {len(hard_negative_indices)} hard negatives out of {np.sum(is_negative)} total negatives."
        )

        # 3. Save
        print(f"[Cache] Saving hard negative indices to {filepath}...")
        np.save(filepath, hard_negative_indices)

        return hard_negative_indices

    @staticmethod
    def train_experts(train_df, hard_negative_indices, val_df):
        """
        Phase 3: Train Expert models on the constructed Expert Dataset.
        Dataset: All Positives + Mined Hard Negatives + Random Anchors (1:1 ratio).
        """
        print("\n[TrainingPipeline] Phase 3: Training Expert Models...")

        # 1. Construct Expert Dataset
        expert_data = DataLoader.prepare_expert_dataset(
            train_df,
            hard_negative_indices,
            anchor_ratio=ANCHOR_RATIO,
            random_state=SEED,
        )

        feature_cols = TrainingPipeline._get_feature_cols(expert_data)
        X_expert = expert_data[feature_cols]
        y_expert = expert_data["contact"]

        X_val = val_df[feature_cols]
        y_val = val_df["contact"]

        models = {}

        # 2. Train LightGBM Expert
        print(" -> Training Expert A (LightGBM)...")
        lgbm_expert = get_model("lgbm", random_state=SEED)
        lgbm_expert.fit(
            X_expert, y_expert, eval_set=[(X_val, y_val)], eval_metric="logloss"
        )
        models["lgbm"] = lgbm_expert
        save_model(lgbm_expert, os.path.join(WORKING_DIR, "models/expert_lgbm.joblib"))

        # 3. Train XGBoost Expert
        print(" -> Training Expert B (XGBoost)...")
        xgb_expert = get_model("xgb", random_state=SEED)
        xgb_expert.fit(X_expert, y_expert, eval_set=[(X_val, y_val)], verbose=False)
        models["xgb"] = xgb_expert
        save_model(xgb_expert, os.path.join(WORKING_DIR, "models/expert_xgb.joblib"))

        return models

    @staticmethod
    def evaluate_and_optimize(expert_models, val_df):
        """
        Evaluates the ensemble on the validation set and optimizes the decision threshold.
        """
        print("\n[TrainingPipeline] Evaluation & Threshold Optimization...")

        feature_cols = TrainingPipeline._get_feature_cols(val_df)
        X_val = val_df[feature_cols]
        y_val = val_df["contact"].values

        # Ensemble Prediction
        p_lgbm = expert_models["lgbm"].predict_proba(X_val)[:, 1]
        p_xgb = expert_models["xgb"].predict_proba(X_val)[:, 1]

        # Unweighted Average
        y_pred_proba = (p_lgbm + p_xgb) / 2.0

        # Optimize Threshold
        best_threshold, best_mcc = optimize_threshold(y_val, y_pred_proba)

        print(f"Validation MCC: {best_mcc}")
        print(f"Optimal Threshold: {best_threshold}")

        # Save threshold
        thresh_path = os.path.join(WORKING_DIR, "models/best_threshold.npy")
        np.save(thresh_path, np.array([best_threshold]))

        return best_threshold

    @staticmethod
    def run_pipeline(load_cached_data=True):
        """
        Main entry point to run the full training pipeline.
        """
        seed_everything(SEED)

        # 1. Load Data
        train_df = DataLoader.load_train_data(load_cached_data=load_cached_data)
        val_df = DataLoader.load_val_data(load_cached_data=load_cached_data)

        # 2. Phase 1: Train Scouts
        # Check if scouts exist to skip training if desired, but for this pipeline we train or overwrite
        scout_models = TrainingPipeline.train_scouts(train_df, val_df)

        # 3. Phase 2: Mine Hard Negatives
        hard_neg_indices = TrainingPipeline.mine_hard_negatives(
            scout_models, train_df, load_cached_data=load_cached_data
        )

        # Clean up scouts to free memory
        del scout_models
        gc.collect()

        # 4. Phase 3: Train Experts
        expert_models = TrainingPipeline.train_experts(
            train_df, hard_neg_indices, val_df
        )

        # 5. Evaluate
        TrainingPipeline.evaluate_and_optimize(expert_models, val_df)

        print("\n[TrainingPipeline] Pipeline Completed Successfully.")
