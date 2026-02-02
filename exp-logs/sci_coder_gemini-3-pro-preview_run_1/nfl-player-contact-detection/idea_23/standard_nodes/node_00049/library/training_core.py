import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.data_manager import DataManager
from library.models import LGBMWrapper, XGBWrapper, HistGBWrapper


class TrainingCore:
    def __init__(self):
        self.config = Config
        self.dm = DataManager()
        self.features = self.config.FEATURES
        # We use the smoothed target for training if available, else raw contact
        self.target_col = "contact_smooth"
        self.raw_target_col = "contact"

        # Setup specific directories for this strategy
        self.base_model_dir = self.config.MODEL_DIR
        self.scout_dir = os.path.join(self.base_model_dir, "scouts")
        self.expert_dir = os.path.join(self.base_model_dir, "experts")
        self.cache_dir = self.config.CACHE_DIR

        os.makedirs(self.scout_dir, exist_ok=True)
        os.makedirs(self.expert_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_target(self, df):
        """Helper to select the correct target column."""
        if self.target_col in df.columns:
            return df[self.target_col]
        return df[self.raw_target_col]

    def train_scouts(self, df_train):
        """
        Phase 1: Train Scout models on a balanced dataset.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # 1. Get Balanced Dataset
        df_scout = self.dm.get_scout_dataset(df_train)
        X = df_scout[self.features]
        y = self._get_target(df_scout)

        print(f"Scout Training Data Shape: {X.shape}")

        # 2. Initialize Models
        models = [LGBMWrapper(), XGBWrapper(), HistGBWrapper()]
        trained_scouts = []

        for model in models:
            # Redirect save path to scouts folder
            model.model_dir = self.scout_dir

            # Fit (No validation set for scouts to maximize diversity on limited balanced data,
            # or we could split df_scout, but standard practice here is fitting on the balanced chunk)
            model.fit(X, y)
            model.save()
            trained_scouts.append(model)

        return trained_scouts

    def mine_hard_negatives(self, df_train, scouts, load_cached_data=True):
        """
        Phase 2: Mine Hard Negatives from the full training set.
        Implements Caching.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")

        cache_path = os.path.join(self.cache_dir, "hard_negative_indices.npy")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached hard negative indices from {cache_path}")
            return np.load(cache_path)

        print("Mining hard negatives from scratch...")

        # 2. Prepare Data
        X_full = df_train[self.features]
        # We only care about negatives
        negative_mask = df_train[self.raw_target_col] == 0

        # Optimization: Only predict on actual negatives to save inference time?
        # However, to keep indices aligned with df_train, we predict on all or map indices carefully.
        # Predicting on all is safer for index alignment.

        # 3. Generate Predictions
        hard_negative_mask = np.zeros(len(df_train), dtype=bool)

        for model in scouts:
            preds = model.predict(X_full)
            # Hard Negative: Ground Truth is 0 AND Prediction > Threshold
            # Note: preds might be probability or binary depending on model,
            # but wrappers return probabilities (except HistGB predict returns class? No, wrapper fixed to return prob).

            # Check wrapper implementation in prompt:
            # HistGBWrapper.predict returns self.model.predict_proba(X)[:, 1] -> Probabilities.
            # LGBM/XGB return probabilities.

            mask = (preds > self.config.HARD_NEGATIVE_THRESHOLD) & negative_mask
            hard_negative_mask = hard_negative_mask | mask

        # 4. Extract Indices
        # We need the index of the dataframe rows that satisfy the condition
        hard_negative_indices = df_train.index[hard_negative_mask].to_numpy()

        print(
            f"Found {len(hard_negative_indices)} hard negatives out of {negative_mask.sum()} total negatives."
        )

        # 5. Save Cache
        print(f"Saving indices to {cache_path}")
        np.save(cache_path, hard_negative_indices)

        return hard_negative_indices

    def train_experts(self, df_train, hard_negative_indices, df_val):
        """
        Phase 3: Train Expert models on the Anchored Dataset.
        """
        print("\n--- Phase 3: Training Experts ---")

        # 1. Construct Anchored Dataset
        df_expert = self.dm.get_anchored_dataset(df_train, hard_negative_indices)
        X_train = df_expert[self.features]
        y_train = self._get_target(df_expert)

        print(f"Expert Training Data Shape: {X_train.shape}")

        # 2. Prepare Validation Data
        X_val = df_val[self.features]
        y_val = self._get_target(
            df_val
        )  # Validation targets might be smoothed or raw. Wrapper handles binarization for metric.

        # 3. Train Models
        models = [LGBMWrapper(), XGBWrapper(), HistGBWrapper()]

        for model in models:
            # Redirect save path to experts folder
            model.model_dir = self.expert_dir

            # Fit with validation
            model.fit(X_train, y_train, X_val, y_val)
            model.save()

    def run(self, load_cached_data=True):
        """
        Executes the full Tri-Scout Anchored Mining Curriculum.
        """
        print("Starting VDAM-E Training Pipeline...")

        # 1. Load Data
        print("Loading Training Features...")
        df_train = self.dm.get_train_features(load_cached_data=load_cached_data)

        print("Loading Validation Features...")
        df_val = self.dm.get_val_features(load_cached_data=load_cached_data)

        # 2. Train Scouts
        scouts = self.train_scouts(df_train)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(
            df_train, scouts, load_cached_data=load_cached_data
        )

        # 4. Train Experts
        self.train_experts(df_train, hard_neg_indices, df_val)

        print("Training Pipeline Completed Successfully.")
