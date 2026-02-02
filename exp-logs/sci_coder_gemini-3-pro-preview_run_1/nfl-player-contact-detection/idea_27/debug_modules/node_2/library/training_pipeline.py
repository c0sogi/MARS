import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import CacheManager, seed_everything, compute_mcc
from library.data_factory import DataFactory
from library.feature_engineering import FeatureEngineer
from library.model_factory import ModelFactory


class TrainingPipeline:
    """
    Orchestrates the Tri-Scout Anchored Mining Curriculum.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        self.data_factory = DataFactory()
        self.feature_engineer = FeatureEngineer()
        seed_everything(Config.SEED)

    def prepare_data(self, load_cached_data=True):
        """
        Loads and processes train and validation features.
        """
        # Load and process Train
        raw_train = self.data_factory.get_data(
            "train", load_cached_data=load_cached_data
        )
        df_train = self.feature_engineer.process_features(
            raw_train, "train", load_cached_data=load_cached_data
        )

        # Load and process Val
        raw_val = self.data_factory.get_data("val", load_cached_data=load_cached_data)
        df_val = self.feature_engineer.process_features(
            raw_val, "val", load_cached_data=load_cached_data
        )

        return df_train, df_val

    def train_scouts(self, df_train, load_cached_models=True):
        """
        Phase 1: Train Scouts on a balanced subset of gated survivors.
        """
        scout_models = {}
        model_names = ["lgbm", "xgb"]

        # Check if all models exist in cache
        all_cached = True
        for name in model_names:
            if not self.cache_manager.exists(f"scout_{name}.joblib"):
                all_cached = False
                break

        if load_cached_models and all_cached:
            print("Loading cached Scout models...")
            for name in model_names:
                scout_models[name] = self.cache_manager.load_joblib(
                    f"scout_{name}.joblib"
                )
            return scout_models

        print("Training Scouts...")

        # Create Balanced Subset for Scouts
        # Positives
        pos_mask = df_train["contact"] == 1
        pos_df = df_train[pos_mask]

        # Random Negatives (1:1 ratio roughly for scouts)
        neg_mask = df_train["contact"] == 0
        neg_df = df_train[neg_mask].sample(n=len(pos_df), random_state=Config.SEED)

        scout_train_df = (
            pd.concat([pos_df, neg_df])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        X = scout_train_df[Config.FEATURE_COLS]
        y = scout_train_df["contact"]

        for name in model_names:
            print(f"Training Scout: {name}")
            model = ModelFactory.get_model(name, epochs=Config.SCOUT_EPOCHS)
            model.fit(X, y)  # Scouts don't strictly need val set, they are for mining

            self.cache_manager.save_joblib(model, f"scout_{name}.joblib")
            scout_models[name] = model

        return scout_models

    def mine_hard_negatives(self, df_train, scout_models, load_cached_data=True):
        """
        Phase 2: Run Scouts on full training set to find Hard Negatives.
        Returns indices of hard negatives.
        """
        cache_file = "hard_negative_indices.npy"

        if load_cached_data and self.cache_manager.exists(cache_file):
            print("Loading cached Hard Negative indices...")
            return self.cache_manager.load_npy(cache_file)

        print("Mining Hard Negatives...")

        # We only care about negatives
        neg_mask = df_train["contact"] == 0
        neg_indices = df_train.index[neg_mask]
        X_neg = df_train.loc[neg_indices, Config.FEATURE_COLS]

        # Union Logic: If ANY scout says prob > Threshold
        hard_mask_local = np.zeros(len(X_neg), dtype=bool)

        for name, model in scout_models.items():
            probs = model.predict_proba(X_neg)[:, 1]
            hard_mask_local |= probs > Config.HARD_NEGATIVE_THRESHOLD

        # Get the global indices of these hard negatives
        hard_negative_indices = neg_indices[hard_mask_local].to_numpy()

        print(
            f"Mined {len(hard_negative_indices)} Hard Negatives out of {len(neg_indices)} total negatives."
        )

        self.cache_manager.save_npy(hard_negative_indices, cache_file)
        return hard_negative_indices

    def apply_label_smoothing(self, df):
        """
        Applies Gaussian smoothing to the contact label over time per pair.
        """
        # We need to ensure sorting by pair and step (should be done in feature engineering, but safety first)
        df = df.sort_values(by=["pair_id", "step"])

        # Define Gaussian Window
        # Sigma = 1.0 step. Window size of ~4*sigma is sufficient (e.g., 5 or 7).
        # win_type='gaussian' requires 'std' arg in mean()

        def smooth_group(x):
            return x.rolling(
                window=7, win_type="gaussian", min_periods=1, center=True
            ).mean(std=Config.LABEL_SMOOTHING_SIGMA)

        # Apply transformation
        # Note: This returns a series indexed by the original df index
        smoothed_targets = df.groupby("pair_id")["contact"].transform(smooth_group)

        # Fill NaNs (if any remain) with original
        smoothed_targets = smoothed_targets.fillna(df["contact"])

        return smoothed_targets

    def train_experts(
        self, df_train, df_val, hard_negative_indices, load_cached_models=True
    ):
        """
        Phase 3: Train Expert Ensemble on Anchored Dataset with Label Smoothing.
        """
        expert_models = {}
        model_names = ["lgbm", "xgb"]

        # Check cache
        all_cached = True
        for name in model_names:
            if not self.cache_manager.exists(f"expert_{name}.joblib"):
                all_cached = False
                break

        if load_cached_models and all_cached:
            print("Loading cached Expert models...")
            for name in model_names:
                expert_models[name] = self.cache_manager.load_joblib(
                    f"expert_{name}.joblib"
                )
            return expert_models

        print("Constructing Anchored Dataset...")

        # 1. Positives
        pos_df = df_train[df_train["contact"] == 1]

        # 2. Hard Negatives
        hard_neg_df = df_train.loc[df_train.index.isin(hard_negative_indices)]

        # 3. Random Anchors (Easy Negatives)
        # Exclude positives and hard negatives
        exclude_mask = (df_train["contact"] == 1) | (
            df_train.index.isin(hard_negative_indices)
        )
        easy_candidates = df_train[~exclude_mask]

        # Calculate number of anchors needed
        n_anchors = int((len(pos_df) + len(hard_neg_df)) * Config.ANCHOR_RATIO)
        # Cap at available candidates
        n_anchors = min(n_anchors, len(easy_candidates))

        anchor_df = easy_candidates.sample(n=n_anchors, random_state=Config.SEED)

        # Combine
        expert_train_df = pd.concat([pos_df, hard_neg_df, anchor_df]).sort_values(
            by=["pair_id", "step"]
        )

        print(
            f"Expert Dataset Size: {len(expert_train_df)} (Pos: {len(pos_df)}, HardNeg: {len(hard_neg_df)}, Anchors: {len(anchor_df)})"
        )

        # Apply Label Smoothing to Targets
        print("Applying Temporal Label Smoothing...")
        y_smooth = self.apply_label_smoothing(expert_train_df)
        X = expert_train_df[Config.FEATURE_COLS]

        # Prepare Validation Data
        X_val = df_val[Config.FEATURE_COLS]
        y_val = df_val["contact"]  # Val targets remain binary for metric calculation

        print("Training Experts...")
        for name in model_names:
            print(f"Training Expert: {name}")
            model = ModelFactory.get_model(name, epochs=Config.EXPERT_EPOCHS)

            # Note: We use smoothed targets for training
            model.fit(
                X,
                y_smooth,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            )

            self.cache_manager.save_joblib(model, f"expert_{name}.joblib")
            expert_models[name] = model

            # Log basic validation metric
            val_probs = model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            mcc = compute_mcc(y_val, val_preds)
            print(f"Expert {name} Val MCC (default thresh): {mcc}")

        return expert_models

    def find_optimal_threshold(self, df_val, expert_models):
        """
        Finds the threshold that maximizes MCC on the validation set using ensemble averaging.
        """
        print("Optimizing Decision Threshold...")
        X_val = df_val[Config.FEATURE_COLS]
        y_val = df_val["contact"].values

        # Ensemble Averaging
        avg_probs = np.zeros(len(X_val))
        for name, model in expert_models.items():
            avg_probs += model.predict_proba(X_val)[:, 1]
        avg_probs /= len(expert_models)

        best_mcc = -1.0
        best_thresh = 0.5

        # Grid Search
        thresholds = np.arange(0.1, 0.9, 0.01)
        for thresh in thresholds:
            preds = (avg_probs >= thresh).astype(int)
            mcc = compute_mcc(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Best Threshold: {best_thresh}")
        print(f"Best Validation MCC: {best_mcc}")

        # Save best threshold
        self.cache_manager.save_npy(np.array([best_thresh]), "best_threshold.npy")

        return best_thresh

    def run(self, debug_sample_size=None):
        """
        Main execution method.
        """
        # 1. Prepare Data
        df_train, df_val = self.prepare_data(load_cached_data=True)

        if debug_sample_size:
            print(f"DEBUG: Subsampling train to {debug_sample_size}")
            df_train = df_train.sample(
                n=debug_sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

        # 2. Train Scouts
        scouts = self.train_scouts(df_train, load_cached_models=True)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(
            df_train, scouts, load_cached_data=True
        )

        # 4. Train Experts
        experts = self.train_experts(
            df_train, df_val, hard_neg_indices, load_cached_models=True
        )

        # 5. Optimize
        best_threshold = self.find_optimal_threshold(df_val, experts)

        return experts, best_threshold
