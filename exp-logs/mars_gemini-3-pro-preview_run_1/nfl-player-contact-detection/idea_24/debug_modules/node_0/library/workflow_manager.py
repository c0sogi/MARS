import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import seed_everything, compute_mcc, save_cache, load_cache
from library.data_loader import DataLoader
from library.mining_strategy import MiningStrategy
from library.model_factory import LGBMExpert, XGBExpert, CatBoostExpert


class WorkflowManager:
    """
    Orchestrates the Vector-Aligned Anchored-Mining Ensemble (VAAM-E) pipeline.
    Manages data loading, the mining curriculum, expert training, threshold optimization,
    and final inference.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Paths for artifacts
        self.expert_lgbm_path = os.path.join(self.models_dir, "expert_lgbm.joblib")
        self.expert_xgb_path = os.path.join(self.models_dir, "expert_xgb.joblib")
        self.expert_cat_path = os.path.join(self.models_dir, "expert_cat.joblib")
        self.threshold_path = os.path.join(self.models_dir, "best_threshold.npy")

    def run_training_phase(self, debug=False, load_cached_data=True):
        """
        Executes the full training pipeline:
        1. Load Data
        2. Train Scouts (Balanced)
        3. Mine Hard Negatives
        4. Construct Anchored Expert Dataset (Smoothed Labels)
        5. Train Experts
        6. Optimize Threshold
        """
        print("Starting Training Phase...")

        # 1. Load Data
        loader = DataLoader()
        df_train = loader.get_train_data(debug=debug, load_cached_data=load_cached_data)
        df_val = loader.get_val_data(debug=debug, load_cached_data=load_cached_data)

        # 2. Initialize Mining Strategy
        miner = MiningStrategy()

        # 3. Phase 1: Train Scouts
        # We check if hard negatives are already cached to potentially skip scout training
        # However, strictly following the curriculum, we train scouts if we need to mine.
        # If hard negatives exist and load_cached_data is True, miner.mine_hard_negatives will load them.
        # But we need scouts object to pass to mine_hard_negatives if cache is missing.

        hard_neg_cache_exists = os.path.exists(Config.CACHE_HARD_NEGATIVES)
        scouts = {}

        if not (load_cached_data and hard_neg_cache_exists):
            scouts = miner.train_scouts(df_train, df_val)
        else:
            print(
                "Skipping Scout training (Hard Negatives cache found and loading enabled)."
            )

        # 4. Phase 2: Mine Hard Negatives
        # If cache exists and load_cached_data=True, this loads from disk.
        # Otherwise uses scouts to predict.
        hard_indices = miner.mine_hard_negatives(
            df_train, scouts, load_cached_data=load_cached_data
        )

        # 5. Phase 3: Construct Anchored Dataset
        X_expert_train, y_expert_train = miner.construct_anchored_dataset(
            df_train, hard_indices, anchor_ratio=1.0
        )

        # Prepare Validation Data for Experts
        # Experts validate on raw binary labels to ensure metric alignment
        X_val = df_val[Config.FEATURES]
        y_val = df_val["contact"]

        # 6. Train Experts
        print("\n--- Training Expert Ensemble ---")

        # Expert A: LightGBM
        print("Training Expert A (LightGBM)...")
        expert_lgbm = LGBMExpert()
        expert_lgbm.fit(X_expert_train, y_expert_train, X_val, y_val)
        expert_lgbm.save(self.expert_lgbm_path)

        # Expert B: XGBoost
        print("Training Expert B (XGBoost)...")
        expert_xgb = XGBExpert()
        expert_xgb.fit(X_expert_train, y_expert_train, X_val, y_val)
        expert_xgb.save(self.expert_xgb_path)

        # Expert C: CatBoost
        print("Training Expert C (CatBoost)...")
        expert_cat = CatBoostExpert()
        expert_cat.fit(X_expert_train, y_expert_train, X_val, y_val)
        expert_cat.save(self.expert_cat_path)

        # 7. Threshold Optimization
        print("\n--- Optimizing Decision Threshold ---")

        # Generate predictions from all experts
        p_lgbm = expert_lgbm.predict(X_val)
        p_xgb = expert_xgb.predict(X_val)
        p_cat = expert_cat.predict(X_val)

        # Ensemble (Unweighted Average)
        p_ensemble = (p_lgbm + p_xgb + p_cat) / 3.0

        # Grid Search
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (p_ensemble >= thresh).astype(int)
            score = compute_mcc(y_val, preds)
            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        print(f"Best Validation MCC: {best_mcc}")
        print(f"Optimal Threshold: {best_thresh}")

        # Save Threshold
        save_cache(np.array([best_thresh]), self.threshold_path)

    def run_inference_phase(self, debug=False, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Load Test Data
        2. Load Experts and Threshold
        3. Generate Predictions
        4. Create Submission File
        """
        print("Starting Inference Phase...")

        # 1. Load Data
        loader = DataLoader()
        df_test = loader.get_test_data(debug=debug, load_cached_data=load_cached_data)
        X_test = df_test[Config.FEATURES]

        # 2. Load Models
        print("Loading Expert Models...")
        if not os.path.exists(self.expert_lgbm_path):
            raise FileNotFoundError(
                "Expert models not found. Run training phase first."
            )

        expert_lgbm = LGBMExpert.load(self.expert_lgbm_path)
        expert_xgb = XGBExpert.load(self.expert_xgb_path)
        expert_cat = CatBoostExpert.load(self.expert_cat_path)

        # Load Threshold
        if os.path.exists(self.threshold_path):
            best_thresh = load_cache(self.threshold_path)[0]
            print(f"Loaded optimal threshold: {best_thresh}")
        else:
            print("Warning: Threshold file not found. Defaulting to 0.5")
            best_thresh = 0.5

        # 3. Generate Predictions
        print("Generating Ensemble Predictions...")
        p_lgbm = expert_lgbm.predict(X_test)
        p_xgb = expert_xgb.predict(X_test)
        p_cat = expert_cat.predict(X_test)

        p_ensemble = (p_lgbm + p_xgb + p_cat) / 3.0

        # Apply Threshold
        final_preds = (p_ensemble >= best_thresh).astype(int)

        # 4. Create Submission
        print("Creating Submission File...")
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": final_preds}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total Predictions: {len(submission)}")
        print(f"Positive Predictions: {submission['contact'].sum()}")
