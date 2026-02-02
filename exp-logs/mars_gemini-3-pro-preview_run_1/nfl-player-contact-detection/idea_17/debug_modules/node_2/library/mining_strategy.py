import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from library.config import Config
from library.data_manager import DataManager
from library.model_factory import LGBMWrapper, XGBWrapper


class MiningStrategy:
    """
    Orchestrates the Dual-Scout Diversity Mining curriculum.
    Responsible for training balanced Scout models and mining hard negatives
    from the full gated survivor pool.
    """

    def __init__(self):
        self.dm = DataManager()

    def train_scouts(self, df_train, load_cached_data=True):
        """
        Phase 1: Dual-Scout Training.
        Trains LightGBM and XGBoost scouts on a balanced subset of the gated training data.

        Args:
            df_train (pd.DataFrame): The full gated training dataframe.
            load_cached_data (bool): If True, attempts to load saved models.

        Returns:
            tuple: (scout_lgbm, scout_xgb) trained model instances.
        """
        # Check if models exist and we want to load them
        lgbm_exists = os.path.exists(Config.MODEL_SCOUT_LGBM_PATH)
        xgb_exists = os.path.exists(Config.MODEL_SCOUT_XGB_PATH)

        if load_cached_data and lgbm_exists and xgb_exists:
            print("Loading cached Scout models...")
            scout_lgbm = LGBMWrapper.load(Config.MODEL_SCOUT_LGBM_PATH)
            scout_xgb = XGBWrapper.load(Config.MODEL_SCOUT_XGB_PATH)
            return scout_lgbm, scout_xgb

        print("Initiating Scout Training...")

        # 1. Construct Balanced Scout Dataset
        # Strategy: All Positives + Equal Random Negatives
        X_scout, y_scout = self.dm.get_scout_dataset(df_train)

        # 2. Split for Internal Validation (Early Stopping)
        # We use a fixed seed for reproducibility and stratify by target
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scout, y_scout, test_size=0.2, random_state=Config.SEED, stratify=y_scout
        )

        # 3. Train Scout A (LightGBM)
        print("\n--- Training Scout A (LightGBM) ---")
        scout_lgbm = LGBMWrapper(mode="scout")
        scout_lgbm.fit(X_tr, y_tr, X_val, y_val)
        scout_lgbm.save(Config.MODEL_SCOUT_LGBM_PATH)

        # 4. Train Scout B (XGBoost)
        print("\n--- Training Scout B (XGBoost) ---")
        scout_xgb = XGBWrapper(mode="scout")
        scout_xgb.fit(X_tr, y_tr, X_val, y_val)
        scout_xgb.save(Config.MODEL_SCOUT_XGB_PATH)

        # Cleanup memory
        del X_scout, y_scout, X_tr, X_val, y_tr, y_val
        gc.collect()

        return scout_lgbm, scout_xgb

    def mine_hard_negatives(
        self, df_train, scout_lgbm, scout_xgb, load_cached_data=True
    ):
        """
        Phase 2: Diversity Mining.
        Runs both scouts on the full gated survivor pool to identify hard negatives.

        Args:
            df_train (pd.DataFrame): The full gated training dataframe.
            scout_lgbm (LGBMWrapper): Trained LightGBM scout.
            scout_xgb (XGBWrapper): Trained XGBoost scout.
            load_cached_data (bool): If True, attempts to load cached indices.

        Returns:
            np.array: Array of indices corresponding to hard negatives in df_train.
        """
        # Check cache for indices
        if load_cached_data:
            indices = self.dm.load_hard_negative_indices()
            if len(indices) > 0:
                print("Loaded cached hard negative indices.")
                return indices

        print("Mining Hard Negatives from full gated training set...")

        # Extract features for the full set
        # We access the internal helper from DataManager to ensure consistency with training data
        feature_cols = self.dm._get_feature_columns(df_train)
        X_full = df_train[feature_cols]

        # Generate Predictions
        print("Generating Scout A (LGBM) predictions...")
        probs_lgbm = scout_lgbm.predict(X_full)

        print("Generating Scout B (XGB) predictions...")
        probs_xgb = scout_xgb.predict(X_full)

        # Identify Hard Negatives
        # Definition: Actual Negative (contact=0) AND (P_lgbm > Thresh OR P_xgb > Thresh)
        # We use the Union of failures to maximize edge-case capture.
        is_negative = df_train["contact"] == 0
        is_hard = (probs_lgbm > Config.HARD_NEGATIVE_THRESHOLD) | (
            probs_xgb > Config.HARD_NEGATIVE_THRESHOLD
        )

        hard_negative_mask = is_negative & is_hard
        hard_negative_indices = df_train.index[hard_negative_mask].to_numpy()

        count = len(hard_negative_indices)
        total_neg = is_negative.sum()
        print(
            f"Mining Complete. Found {count} hard negatives out of {total_neg} total negatives ({count/total_neg:.4%})."
        )

        # Save indices for Expert training phase
        self.dm.save_hard_negative_indices(hard_negative_indices)

        # Cleanup
        del X_full, probs_lgbm, probs_xgb
        gc.collect()

        return hard_negative_indices

    def execute(self, df_train, load_cached_data=True):
        """
        Executes the full mining pipeline: Train Scouts -> Mine Negatives.

        Args:
            df_train (pd.DataFrame): The full gated training dataframe.
            load_cached_data (bool): Whether to use cached models/indices.

        Returns:
            np.array: Array of hard negative indices.
        """
        # 1. Train Scouts
        scout_lgbm, scout_xgb = self.train_scouts(
            df_train, load_cached_data=load_cached_data
        )

        # 2. Mine Hard Negatives
        hard_indices = self.mine_hard_negatives(
            df_train, scout_lgbm, scout_xgb, load_cached_data=load_cached_data
        )

        return hard_indices
