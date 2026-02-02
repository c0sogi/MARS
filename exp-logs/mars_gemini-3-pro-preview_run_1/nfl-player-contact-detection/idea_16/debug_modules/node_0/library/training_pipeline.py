import pandas as pd
import numpy as np
import os
import joblib
from library.config import PathConfig, ModelConfig, FeatureConfig
from library.utils import seed_everything, optimize_mcc_threshold, CacheManager
from library.data_loader import DataLoader
from library.model_factory import LGBMWrapper, XGBWrapper


class TrainingPipeline:
    """
    Orchestrates the Dual-Scout Physics-Enhanced Mining Ensemble (DSP-EME) pipeline.
    Manages the lifecycle of data loading, scout training, hard negative mining,
    expert training, and submission generation.
    """

    def __init__(self):
        self.paths = PathConfig()
        self.model_config = ModelConfig()
        self.feature_config = FeatureConfig()
        self.loader = DataLoader()
        self.cache = CacheManager()

        # Ensure reproducibility
        seed_everything(self.model_config.SEED)

    def train_scouts(self, train_df):
        """
        Stage 1: Trains heterogeneous Scout models (LGBM + XGB) on a balanced dataset.
        The goal is to create models capable of identifying potential contacts and
        filtering out obvious non-contacts.
        """
        print("\n--- Stage 1: Training Scout Models ---")

        # 1. Construct Balanced Dataset (Positives + Equal Random Negatives)
        pos_df = train_df[train_df["contact"] == 1]
        neg_df = train_df[train_df["contact"] == 0]

        # Sample negatives to match positives
        if len(neg_df) > len(pos_df):
            neg_sample = neg_df.sample(
                n=len(pos_df), random_state=self.model_config.SEED
            )
        else:
            neg_sample = neg_df

        scout_train_df = (
            pd.concat([pos_df, neg_sample])
            .sample(frac=1.0, random_state=self.model_config.SEED)
            .reset_index(drop=True)
        )

        print(
            f"Scout Training Data: {len(scout_train_df)} rows "
            f"(Pos: {len(pos_df)}, Neg: {len(neg_sample)})"
        )

        X_scout, y_scout = self.loader.split_features_target(scout_train_df)

        # 2. Train Scout A (LightGBM)
        print("Training Scout A (LightGBM)...")
        scout_lgbm = LGBMWrapper(self.model_config.SCOUT_LGBM_PARAMS)
        scout_lgbm.fit(X_scout, y_scout)
        scout_lgbm.save(self.paths.SCOUT_LGBM_PATH)

        # 3. Train Scout B (XGBoost)
        print("Training Scout B (XGBoost)...")
        scout_xgb = XGBWrapper(self.model_config.SCOUT_XGB_PARAMS)
        scout_xgb.fit(X_scout, y_scout)
        scout_xgb.save(self.paths.SCOUT_XGB_PATH)

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(self, train_df, scout_lgbm, scout_xgb, load_cached=True):
        """
        Stage 2: Hard Negative Mining.
        Uses the trained Scouts to scan the entire gated training set.
        Identifies negative samples where either scout predicts a probability > threshold.

        Implements caching for the resulting indices.
        """
        print("\n--- Stage 2: Mining Hard Negatives ---")

        # Check cache
        if load_cached and self.cache.exists(self.paths.HARD_NEGATIVE_INDICES_PATH):
            print("Loading hard negative indices from cache...")
            return self.cache.load_numpy(self.paths.HARD_NEGATIVE_INDICES_PATH)

        print("Mining hard negatives from scratch...")

        # Prepare features for full training set
        X_full, y_full = self.loader.split_features_target(train_df)

        # Get predictions from both scouts
        print("Scoring full dataset with Scout A...")
        preds_lgbm = scout_lgbm.predict(X_full)

        print("Scoring full dataset with Scout B...")
        preds_xgb = scout_xgb.predict(X_full)

        # Identify Hard Negatives
        # Criteria: True Label is 0 AND (ScoutA > Threshold OR ScoutB > Threshold)
        threshold = self.model_config.HARD_NEGATIVE_THRESHOLD
        is_negative = (y_full == 0).values
        is_hard = (preds_lgbm > threshold) | (preds_xgb > threshold)

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = train_df.index[hard_negative_mask].to_numpy()

        print(
            f"Mined {len(hard_negative_indices)} hard negatives from {len(train_df)} total rows."
        )

        # Save to cache
        self.cache.save_numpy(
            hard_negative_indices, self.paths.HARD_NEGATIVE_INDICES_PATH
        )

        return hard_negative_indices

    def train_experts(self, train_df, val_df, hard_neg_indices):
        """
        Stage 3: Train Expert Ensemble.
        Constructs a high-information-density dataset:
        - All Positives
        - All Mined Hard Negatives
        - Small Buffer of Random Negatives (to anchor decision boundary)

        Trains high-capacity LGBM and XGB models and validates them.
        """
        print("\n--- Stage 3: Training Expert Models ---")

        # 1. Construct Expert Dataset
        pos_df = train_df[train_df["contact"] == 1]

        # Retrieve hard negatives by index
        hard_neg_df = train_df.loc[hard_neg_indices]

        # Create a buffer of "Easy" Negatives
        # Exclude hard negatives and positives to find the pool of easy negatives
        # Note: train_df indices are unique per row if reset_index was called in loader,
        # but let's be safe. We assume indices align with hard_neg_indices.

        # Get indices of positives and hard negatives
        pos_indices = set(pos_df.index)
        hard_indices = set(hard_neg_indices)
        used_indices = pos_indices.union(hard_indices)

        # Identify remaining indices (Easy Negatives)
        all_indices = set(train_df.index)
        easy_indices = list(all_indices - used_indices)

        # Sample buffer (e.g., equal to number of positives)
        buffer_size = len(pos_df)
        if len(easy_indices) > buffer_size:
            np.random.seed(self.model_config.SEED)
            buffer_indices = np.random.choice(
                easy_indices, size=buffer_size, replace=False
            )
            buffer_neg_df = train_df.loc[buffer_indices]
        else:
            buffer_neg_df = train_df.loc[easy_indices]

        # Combine
        expert_train_df = pd.concat([pos_df, hard_neg_df, buffer_neg_df])
        expert_train_df = expert_train_df.sample(
            frac=1.0, random_state=self.model_config.SEED
        ).reset_index(drop=True)

        print(f"Expert Training Data: {len(expert_train_df)} rows")
        print(f"  Positives: {len(pos_df)}")
        print(f"  Hard Negatives: {len(hard_neg_df)}")
        print(f"  Buffer Negatives: {len(buffer_neg_df)}")

        X_train, y_train = self.loader.split_features_target(expert_train_df)
        X_val, y_val = self.loader.split_features_target(val_df)

        # 2. Train Expert A (LightGBM)
        print("Training Expert A (LightGBM)...")
        expert_lgbm = LGBMWrapper(self.model_config.EXPERT_LGBM_PARAMS)
        expert_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        expert_lgbm.save(self.paths.EXPERT_LGBM_PATH)

        # 3. Train Expert B (XGBoost)
        print("Training Expert B (XGBoost)...")
        expert_xgb = XGBWrapper(self.model_config.EXPERT_XGB_PARAMS)
        expert_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        expert_xgb.save(self.paths.EXPERT_XGB_PATH)

        # 4. Optimize Ensemble Threshold on Validation Set
        print("Optimizing Ensemble Threshold...")
        p_val_lgbm = expert_lgbm.predict(X_val)
        p_val_xgb = expert_xgb.predict(X_val)
        p_val_ensemble = (p_val_lgbm + p_val_xgb) / 2.0

        best_thresh, best_mcc = optimize_mcc_threshold(y_val, p_val_ensemble)

        # Save threshold for inference
        np.save(
            os.path.join(self.paths.WORKING_DIR, "best_threshold.npy"),
            np.array([best_thresh]),
        )

        return expert_lgbm, expert_xgb, best_thresh

    def generate_submission(self, expert_lgbm, expert_xgb, threshold):
        """
        Generates predictions for the test set using the Expert Ensemble
        and the optimized threshold. Saves to submission.csv.
        """
        print("\n--- Generating Submission ---")

        # Load Test Data
        test_df = self.loader.prepare_test_dataset(load_cached=True)
        X_test, _ = self.loader.split_features_target(test_df)

        # Predict
        print("Predicting with Expert A...")
        p_test_lgbm = expert_lgbm.predict(X_test)

        print("Predicting with Expert B...")
        p_test_xgb = expert_xgb.predict(X_test)

        # Ensemble
        p_test_ensemble = (p_test_lgbm + p_test_xgb) / 2.0

        # Threshold
        predictions = (p_test_ensemble >= threshold).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": test_df["contact_id"], "contact": predictions}
        )

        # Save
        print(f"Saving submission to {self.paths.SUBMISSION_FILE_PATH}...")
        submission.to_csv(self.paths.SUBMISSION_FILE_PATH, index=False)
        print("Submission saved successfully.")

    def run(self, sample_fraction=None):
        """
        Main execution entry point.
        """
        # 1. Load Data
        print("Loading Train/Val Data...")
        train_df, val_df = self.loader.prepare_train_val_dataset(
            load_cached=True, sample_fraction=sample_fraction
        )

        # 2. Train Scouts
        scout_lgbm, scout_xgb = self.train_scouts(train_df)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(
            train_df, scout_lgbm, scout_xgb, load_cached=True
        )

        # 4. Train Experts & Validate
        expert_lgbm, expert_xgb, best_threshold = self.train_experts(
            train_df, val_df, hard_neg_indices
        )

        # 5. Generate Submission
        self.generate_submission(expert_lgbm, expert_xgb, best_threshold)

        print("\nPipeline Completed Successfully.")
