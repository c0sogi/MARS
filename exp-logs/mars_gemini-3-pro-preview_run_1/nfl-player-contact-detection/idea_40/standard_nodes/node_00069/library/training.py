import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.models import LGBMWrapper, XGBWrapper, Ensemble
from library.data_manager import DataManager


class Trainer:
    """
    Orchestrates the Dual-Scout Anchored Mining training pipeline.
    """

    def __init__(self, config=Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        seed_everything(self.config.SEED)

    def train_scouts(self, df_train):
        """
        Phase 1: Train Scout models on a balanced dataset.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # Build balanced dataset
        df_scout = self.data_manager.build_scout_dataset(df_train)
        X_scout, y_scout = self.data_manager.get_X_y(df_scout)

        print(f"Scout Training Data Shape: {X_scout.shape}")

        scouts = []

        # Train LightGBM Scout
        lgbm_scout = LGBMWrapper()
        print("Training LightGBM Scout...")
        lgbm_scout.fit(X_scout, y_scout)
        lgbm_scout.save(os.path.join(self.models_dir, "scout_lgbm.joblib"))
        scouts.append(lgbm_scout)

        # Train XGBoost Scout
        xgb_scout = XGBWrapper()
        print("Training XGBoost Scout...")
        xgb_scout.fit(X_scout, y_scout)
        xgb_scout.save(os.path.join(self.models_dir, "scout_xgb.joblib"))
        scouts.append(xgb_scout)

        return scouts

    def mine_hard_negatives(self, df_train, scout_models, load_cached=True):
        """
        Phase 2: Mine Hard Negatives using Scout models.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")
        # Delegate to DataManager which handles logic and caching
        indices = self.data_manager.mine_hard_negatives(
            df_train, scout_models, load_cached_indices=load_cached
        )
        return indices

    def train_experts(self, df_train, hard_negative_indices, df_val):
        """
        Phase 3: Train Expert models on the enriched dataset (Positives + Hard Negatives + Anchors).
        """
        print("\n--- Phase 3: Training Experts ---")

        # Build Expert Dataset
        df_expert = self.data_manager.build_expert_dataset(
            df_train, hard_negative_indices
        )
        X_expert, y_expert = self.data_manager.get_X_y(df_expert)

        # Prepare Validation Data
        X_val, y_val = self.data_manager.get_X_y(df_val)

        print(f"Expert Training Data Shape: {X_expert.shape}")
        print(f"Validation Data Shape: {X_val.shape}")

        experts = []

        # Train LightGBM Expert
        lgbm_expert = LGBMWrapper()
        print("Training LightGBM Expert...")
        lgbm_expert.fit(X_expert, y_expert, X_val, y_val)
        lgbm_expert.save(os.path.join(self.models_dir, "expert_lgbm.joblib"))
        experts.append(lgbm_expert)

        # Train XGBoost Expert
        xgb_expert = XGBWrapper()
        print("Training XGBoost Expert...")
        xgb_expert.fit(X_expert, y_expert, X_val, y_val)
        xgb_expert.save(os.path.join(self.models_dir, "expert_xgb.joblib"))
        experts.append(xgb_expert)

        return experts

    def optimize_threshold(self, expert_models, df_val):
        """
        Phase 4: Optimize Decision Threshold on Validation Set maximizing MCC.
        """
        print("\n--- Phase 4: Optimizing Threshold ---")

        X_val, y_val = self.data_manager.get_X_y(df_val)

        # Create Ensemble
        ensemble = Ensemble(expert_models)

        # Get Probabilities
        print("Generating validation predictions...")
        y_pred_prob = ensemble.predict(X_val)

        # Apply Gating Mask if available (forces 0 for gated rows)
        if "gating_active" in df_val.columns:
            y_pred_prob = y_pred_prob * df_val["gating_active"].values

        # Grid Search for Threshold
        thresholds = np.linspace(0.01, 0.99, 99)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred_binary = (y_pred_prob >= thresh).astype(int)
            mcc = matthews_corrcoef(y_val, y_pred_binary)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Best Validation MCC: {best_mcc}")
        print(f"Optimal Threshold: {best_thresh}")

        # Save threshold
        thresh_path = os.path.join(self.models_dir, "best_threshold.npy")
        np.save(thresh_path, np.array([best_thresh]))

        return best_thresh

    def generate_submission(self, expert_models, threshold, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("\n--- Generating Submission ---")

        # Load Test Data
        df_test = self.data_manager.get_test_data(load_cached_data=load_cached_data)

        # Prepare Features
        # Note: get_X_y strips metadata, but we need contact_id for submission
        X_test, _ = self.data_manager.get_X_y(df_test)

        # Create Ensemble
        ensemble = Ensemble(expert_models)

        # Predict
        print(f"Predicting on {len(X_test)} test samples...")
        probs = ensemble.predict(X_test)

        # Apply Threshold
        predictions = (probs >= threshold).astype(int)

        # Create Submission DataFrame
        # We assume df_test aligns with sample_submission rows because we loaded from test_metadata
        # which was derived from sample_submission.
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Save
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

    def run(self, load_cached_data=True):
        """
        Executes the full training pipeline.
        """
        setup_logging()

        # 1. Load Data
        print("Loading Training and Validation Data...")
        df_train = self.data_manager.get_train_data(load_cached_data=load_cached_data)
        df_val = self.data_manager.get_val_data(load_cached_data=load_cached_data)

        # 2. Train Scouts
        scout_models = self.train_scouts(df_train)

        # 3. Mine Hard Negatives
        # We pass load_cached=load_cached_data to control caching of indices
        hard_neg_indices = self.mine_hard_negatives(
            df_train, scout_models, load_cached=load_cached_data
        )

        # 4. Train Experts
        expert_models = self.train_experts(df_train, hard_neg_indices, df_val)

        # 5. Optimize Threshold
        best_threshold = self.optimize_threshold(expert_models, df_val)

        # 6. Generate Submission
        self.generate_submission(
            expert_models, best_threshold, load_cached_data=load_cached_data
        )

        print("\nPipeline Completed Successfully.")
