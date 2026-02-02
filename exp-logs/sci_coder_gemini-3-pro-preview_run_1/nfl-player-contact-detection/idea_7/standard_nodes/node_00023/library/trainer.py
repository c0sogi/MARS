import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from library.config import (
    LGBM_SCOUT_PARAMS,
    LGBM_EXPERT_PARAMS,
    XGB_EXPERT_PARAMS,
    SCOUT_THRESHOLD,
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import setup_logger, seed_everything
from library.data_loader import DataLoader
from library.features import FeatureFactory
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper


class CascadeTrainer:
    """
    Orchestrates the Tiered Cascade Training Curriculum:
    1. Train Scout (Tier 1 features, Balanced Data)
    2. Mine Hard Negatives (Scout Inference on Full Train)
    3. Train Experts (Tier 2 features, Mined Data)
    4. Inference & Submission
    """

    def __init__(self):
        self.logger = setup_logger()
        self.loader = DataLoader()
        self.factory = FeatureFactory()
        seed_everything(SEED)
        self.models = {}
        self.best_threshold = 0.5

    def get_labels(self, df):
        """Extracts labels from the dataframe."""
        if "contact" in df.columns:
            return df["contact"].values
        return None

    def train_scout(self, df_train, n_rows=None):
        """
        Phase 1: Train the Scout Model.
        Uses Tier 2 (Windowed) features on a balanced subset to ensure temporal context.
        Cite solution_lesson_node_00022: Filtering stage must share temporal receptive field.
        """
        self.logger.info("\n--- Phase 1: Training Scout Model ---")

        # 1. Compute Tier 2 Features (Windowed)
        # If n_rows is specified for debugging, slice df_train
        if n_rows is not None:
            df_train_subset = df_train.iloc[:n_rows].copy()
        else:
            df_train_subset = df_train

        # Use Tier 2 features for Scout to capture temporal dynamics
        X_scout = self.factory.compute_tier2_features(df_train_subset)
        y = self.get_labels(df_train_subset)

        # 2. Balance Dataset (All Positives + Equal number of Random Negatives)
        pos_mask = y == 1
        neg_mask = y == 0

        n_pos = np.sum(pos_mask)
        n_neg_sample = n_pos * 1  # 1:1 Ratio for initial Scout training

        # Get indices
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        # Randomly sample negatives
        np.random.seed(SEED)
        neg_indices_sampled = np.random.choice(
            neg_indices, size=min(n_neg_sample, len(neg_indices)), replace=False
        )

        train_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(train_indices)

        X_balanced = X_scout.iloc[train_indices]
        y_balanced = y[train_indices]

        self.logger.info(
            f"Scout Training Data: {len(X_balanced)} rows (Pos: {n_pos}, Neg: {len(neg_indices_sampled)})"
        )

        # 3. Train Scout
        scout = LGBMClassifierWrapper(LGBM_SCOUT_PARAMS, name="scout_lgbm")
        scout.fit(X_balanced, y_balanced)

        self.models["scout"] = scout

        # Save
        scout.save(os.path.join(WORKING_DIR, "scout_model.joblib"))

        # Cleanup
        del X_scout, X_balanced, y_balanced
        gc.collect()

        return scout

    def mine_hard_negatives(self, df_train, scout_model):
        """
        Phase 2: Hard Negative Mining.
        Runs Scout inference on the full training set to find candidates.
        Returns a boolean mask of rows to keep (Positives + Hard Negatives).
        """
        self.logger.info("\n--- Phase 2: Mining Hard Negatives ---")

        # Compute Tier 2 features for ALL data (Scout now uses temporal features)
        X_mining = self.factory.compute_tier2_features(df_train)
        y = self.get_labels(df_train)

        # Predict
        self.logger.info("Running Scout inference on full training set...")
        probs = scout_model.predict_proba(X_mining)

        # Filter: Keep if Contact=1 OR Prob > Threshold
        # This ensures we never lose ground truth positives
        is_positive = y == 1
        is_hard_negative = probs > SCOUT_THRESHOLD

        mask = is_positive | is_hard_negative

        n_selected = np.sum(mask)
        self.logger.info(
            f"Mining Complete. Selected {n_selected}/{len(df_train)} rows ({n_selected/len(df_train):.2%})"
        )
        self.logger.info(
            f"Breakdown: {np.sum(is_positive)} Positives, {np.sum(is_hard_negative & ~is_positive)} Hard Negatives"
        )

        del X_mining, probs
        gc.collect()

        return mask

    def train_experts(self, df_train, train_mask, df_val):
        """
        Phase 3: Train Expert Ensemble.
        Computes Tier 2 features for mined training data and full validation data.
        Trains LGBM and XGB models.
        """
        self.logger.info("\n--- Phase 3: Training Expert Ensemble ---")

        # 1. Prepare Training Data (Mined)
        self.logger.info("Generating Tier 2 features for MINED training data...")
        X_train = self.factory.compute_tier2_features(df_train, target_mask=train_mask)
        y_train = self.get_labels(df_train)[train_mask]

        # 2. Prepare Validation Data (Full)
        # We validate on the full set to ensure the model generalizes to easy negatives too
        # (or at least we get accurate metrics).
        self.logger.info("Generating Tier 2 features for FULL validation data...")
        X_val = self.factory.compute_tier2_features(df_val)
        y_val = self.get_labels(df_val)

        self.logger.info(
            f"Expert Train Shape: {X_train.shape}, Val Shape: {X_val.shape}"
        )

        # 3. Train LightGBM Expert
        lgbm_expert = LGBMClassifierWrapper(LGBM_EXPERT_PARAMS, name="expert_lgbm")
        lgbm_expert.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=100
        )
        self.models["expert_lgbm"] = lgbm_expert
        lgbm_expert.save(os.path.join(WORKING_DIR, "expert_lgbm.joblib"))

        # 4. Train XGBoost Expert
        xgb_expert = XGBClassifierWrapper(XGB_EXPERT_PARAMS, name="expert_xgb")
        xgb_expert.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=100
        )
        self.models["expert_xgb"] = xgb_expert
        xgb_expert.save(os.path.join(WORKING_DIR, "expert_xgb.joblib"))

        # 5. Optimize Threshold on Validation
        self.logger.info("Optimizing Decision Threshold...")
        p_val_lgbm = lgbm_expert.predict_proba(X_val)
        p_val_xgb = xgb_expert.predict_proba(X_val)
        p_val_ens = 0.5 * p_val_lgbm + 0.5 * p_val_xgb

        best_mcc = -1
        best_thresh = 0.5

        # Search range
        thresholds = np.arange(0.1, 0.9, 0.05)
        for t in thresholds:
            preds = (p_val_ens >= t).astype(int)
            mcc = matthews_corrcoef(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        self.best_threshold = best_thresh
        self.logger.info(
            f"Best Validation MCC: {best_mcc:.10f} at Threshold: {best_thresh:.2f}"
        )

        # Cleanup
        del X_train, y_train, X_val, y_val
        gc.collect()

    def generate_submission(self, df_test):
        """
        Generates predictions for the test set and saves the submission file.
        """
        self.logger.info("\n--- Phase 4: Inference & Submission ---")

        # 1. Compute Features (Full Test Set)
        # Note: Test set is small enough to run Tier 2 on everything
        X_test = self.factory.compute_tier2_features(df_test)

        # 2. Predict
        lgbm_expert = self.models["expert_lgbm"]
        xgb_expert = self.models["expert_xgb"]

        p_test_lgbm = lgbm_expert.predict_proba(X_test)
        p_test_xgb = xgb_expert.predict_proba(X_test)

        p_test_ens = 0.5 * p_test_lgbm + 0.5 * p_test_xgb

        # 3. Threshold
        predictions = (p_test_ens >= self.best_threshold).astype(int)

        # 4. Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # 5. Save
        self.logger.info(f"Saving submission to {SUBMISSION_PATH}")
        sub_df.to_csv(SUBMISSION_PATH, index=False)

        # Print stats
        pos_ratio = predictions.mean()
        self.logger.info(f"Test Positive Ratio: {pos_ratio:.4f}")

    def run(self, debug_rows=None):
        """
        Main execution flow.
        """
        # Load Data
        self.logger.info("Loading Base Tables...")
        df_train = self.loader.prepare_base_table(mode="train", n_rows=debug_rows)
        df_val = self.loader.prepare_base_table(mode="val", n_rows=debug_rows)

        # Phase 1: Scout
        scout = self.train_scout(df_train, n_rows=debug_rows)

        # Phase 2: Mining
        mining_mask = self.mine_hard_negatives(df_train, scout)

        # Phase 3: Experts
        self.train_experts(df_train, mining_mask, df_val)

        # Phase 4: Inference
        # Load test data
        df_test = self.loader.prepare_base_table(mode="test")
        self.generate_submission(df_test)

        self.logger.info("Pipeline Completed Successfully.")
