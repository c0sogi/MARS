import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef, log_loss

from library.config import Config
from library.utils import setup_logging, Timer, seed_everything
from library.data_loader import DataLoader
from library.models import LGBMModel, XGBModel, EnsemblePredictor


class Trainer:
    """
    Orchestrates the Vector-Decomposed Physics Ensemble training pipeline.
    Manages Dual-Scout mining, Temporal Label Smoothing, and Ensemble training.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.working_dir = Config.WORKING_DIR
        self.scout_dir = os.path.join(self.working_dir, "models", "scouts")
        self.expert_dir = os.path.join(self.working_dir, "models", "experts")
        self.cache_dir = os.path.join(self.working_dir, "cache")

        for d in [self.scout_dir, self.expert_dir, self.cache_dir]:
            os.makedirs(d, exist_ok=True)

        setup_logging()
        seed_everything(Config.SEED)

    def apply_temporal_smoothing(self, df):
        """
        Applies Temporal Gaussian Smoothing to contact labels.
        Sigma = 1.0 step (0.1s).

        Vectorized implementation using shift/merge.
        """
        with Timer("Temporal Label Smoothing"):
            # Keys to identify unique pairs
            keys = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

            # Create soft target column, init with hard labels
            df["contact_soft"] = df["contact"].astype(float)

            # Gaussian weights for sigma=1.0
            # t=0: 1.0
            # t=1: exp(-0.5) ~ 0.6065
            # t=2: exp(-2.0) ~ 0.1353
            offsets = [-2, -1, 1, 2]
            sigma = Config.SMOOTHING_SIGMA

            # Filter only positives to propagate
            positives = df[df["contact"] == 1][keys + ["step"]].copy()
            positives["is_positive"] = 1.0

            for offset in offsets:
                weight = np.exp(-0.5 * (offset / sigma) ** 2)

                # Shift positives to simulate neighbor influence
                # If we want to affect t+offset, the source is at t.
                # So if we are at t_target, we look for source at t_target - offset.
                # Here we simply shift the step of the positive source to match the target.
                # Source at T affects Target at T+offset.
                # So we label the source row as being at "step + offset".

                temp = positives.copy()
                temp["step"] = temp["step"] + offset
                temp["weight"] = weight

                # Merge back to full df
                # Left join ensures we only update existing rows (gated survivors)
                df = df.merge(
                    temp, on=keys + ["step"], how="left", suffixes=("", "_smooth")
                )

                # Update soft contact: max(current, new_weight)
                # If 'weight' column is NaN, it means no positive neighbor at this offset
                mask = df["weight"].notna()
                if mask.sum() > 0:
                    df.loc[mask, "contact_soft"] = np.maximum(
                        df.loc[mask, "contact_soft"], df.loc[mask, "weight"]
                    )

                # Cleanup merge artifacts
                df = df.drop(
                    columns=[
                        "is_positive",
                        "weight",
                        "is_positive_smooth",
                        "weight_smooth",
                    ],
                    errors="ignore",
                )

            # Replace original contact with soft labels for training
            # We keep 'contact' as binary for reference if needed, but models use 'contact' usually.
            # To avoid changing model code which expects 'contact' column as target:
            df["contact_hard"] = df["contact"]  # Backup
            df["contact"] = df["contact_soft"]

            print(f"Smoothing Complete. Soft label mean: {df['contact'].mean():.4f}")

        return df

    def train_scouts(self, sample_size=None):
        """
        Phase 1: Train Scout Models (LGBM, XGB) on balanced data.
        """
        print("\n=== Phase 1: Training Scouts ===")

        # 1. Load Data
        df_train = self.loader.load_train_data(
            load_cached_data=True, sample_size=sample_size
        )

        # 2. Balance Data
        # Keep all positives
        pos_mask = df_train["contact"] == 1
        neg_mask = df_train["contact"] == 0

        df_pos = df_train[pos_mask]
        df_neg = df_train[neg_mask]

        n_pos = len(df_pos)
        n_neg_keep = int(n_pos * Config.INITIAL_NEG_POS_RATIO)

        if len(df_neg) > n_neg_keep:
            df_neg = df_neg.sample(n=n_neg_keep, random_state=Config.SEED)

        df_balanced = (
            pd.concat([df_pos, df_neg])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )
        print(
            f"Balanced Scout Dataset: {len(df_balanced)} rows (Pos: {len(df_pos)}, Neg: {len(df_neg)})"
        )

        # 3. Train Scouts
        # We don't use validation set for scouts to save time/complexity,
        # or we could use a split of the balanced set.
        # Given the pipeline, we'll train on full balanced set.

        X = df_balanced
        y = df_balanced["contact"]

        # Scout A: LGBM
        scout_lgbm = LGBMModel()
        scout_lgbm.fit(X, y)
        scout_lgbm.save(self.scout_dir)

        # Scout B: XGB
        scout_xgb = XGBModel()
        scout_xgb.fit(X, y)
        scout_xgb.save(self.scout_dir)

        # Cleanup
        del df_train, df_balanced, df_pos, df_neg, X, y
        gc.collect()

    def mine_hard_negatives(self, load_cached_data=True, sample_size=None):
        """
        Phase 2: Mine Hard Negatives using Scouts.
        Returns the Expert Dataset (Positives + Hard Negatives).
        """
        print("\n=== Phase 2: Mining Hard Negatives ===")

        cache_file = os.path.join(self.cache_dir, "hard_negative_indices.npy")

        # 1. Load Full Data
        df_train = self.loader.load_train_data(
            load_cached_data=True, sample_size=sample_size
        )

        # Check cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading hard negative indices from {cache_file}...")
            hard_neg_indices = np.load(cache_file)
        else:
            print("Running Scouts on full training set...")

            # Load Scouts
            scout_lgbm = LGBMModel()
            scout_lgbm.load(self.scout_dir)

            scout_xgb = XGBModel()
            scout_xgb.load(self.scout_dir)

            # Predict
            # We only need to predict on Negatives to find hard ones,
            # but predicting all is easier for indexing.
            neg_mask = df_train["contact"] == 0
            X_neg = df_train[neg_mask]

            if len(X_neg) == 0:
                print(
                    "No negatives found in dataset (likely small sample). Skipping mining."
                )
                hard_neg_indices = np.array([])
            else:
                p_lgbm = scout_lgbm.predict_proba(X_neg)
                p_xgb = scout_xgb.predict_proba(X_neg)

                # Union of hard negatives
                is_hard = (p_lgbm > Config.HARD_MINING_THRESHOLD) | (
                    p_xgb > Config.HARD_MINING_THRESHOLD
                )

                # Get original indices
                hard_neg_indices = X_neg.index[is_hard].values

                print(
                    f"Mined {len(hard_neg_indices)} Hard Negatives out of {len(X_neg)} Negatives."
                )

                # Save Cache
                np.save(cache_file, hard_neg_indices)

        # Construct Expert Dataset
        pos_indices = df_train[df_train["contact"] == 1].index.values

        # Combine indices
        expert_indices = np.union1d(pos_indices, hard_neg_indices)

        df_expert = (
            df_train.loc[expert_indices]
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )
        print(f"Expert Dataset Constructed: {len(df_expert)} rows.")

        return df_expert

    def train_experts(self, sample_size=None):
        """
        Phase 3: Train Expert Ensemble with Temporal Label Smoothing.
        """
        print("\n=== Phase 3: Training Expert Ensemble ===")

        # 1. Get Data
        df_expert = self.mine_hard_negatives(
            load_cached_data=True, sample_size=sample_size
        )

        # 2. Apply Smoothing
        df_expert = self.apply_temporal_smoothing(df_expert)

        # 3. Load Validation Data
        df_val = self.loader.load_val_data(
            load_cached_data=True, sample_size=sample_size
        )

        # 4. Train Ensemble
        # Note: df_expert['contact'] is now soft (float).
        # df_val['contact'] is hard (0/1).

        X_train = df_expert
        y_train = df_expert["contact"]

        X_val = df_val
        y_val = df_val["contact"]

        ensemble = EnsemblePredictor()
        ensemble.train(X_train, y_train, X_val, y_val)

        # 5. Save
        ensemble.save_models(self.expert_dir)

        # Cleanup
        del df_expert, df_val, X_train, y_train, X_val, y_val
        gc.collect()

    def optimize_threshold(self, sample_size=None):
        """
        Phase 4: Threshold Optimization on Validation Set.
        """
        print("\n=== Phase 4: Threshold Optimization ===")

        # Load Val
        df_val = self.loader.load_val_data(
            load_cached_data=True, sample_size=sample_size
        )

        # Load Ensemble
        ensemble = EnsemblePredictor()
        ensemble.load_models(self.expert_dir)

        # Evaluate
        # Note: evaluate method in EnsemblePredictor prints metrics and returns best threshold
        best_threshold = ensemble.evaluate(df_val, df_val["contact"])

        # Save Threshold
        thresh_path = os.path.join(self.working_dir, "best_threshold.npy")
        np.save(thresh_path, np.array([best_threshold]))
        print(f"Best Threshold {best_threshold:.4f} saved to {thresh_path}")

    def generate_submission(self):
        """
        Phase 5: Inference on Test Set.
        """
        print("\n=== Phase 5: Generating Submission ===")

        # 1. Load Test Data
        # load_test_data skips gating, so we get all required rows
        df_test = self.loader.load_test_data(load_cached_data=True)

        # 2. Load Models and Threshold
        ensemble = EnsemblePredictor()
        ensemble.load_models(self.expert_dir)

        thresh_path = os.path.join(self.working_dir, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = float(np.load(thresh_path)[0])
        else:
            print("Warning: Threshold file not found. Using default 0.5")
            threshold = 0.5

        print(f"Using Threshold: {threshold:.4f}")

        # 3. Predict
        # We use predict_proba and apply threshold manually to be safe
        probs = ensemble.predict_proba(df_test)
        preds = (probs >= threshold).astype(int)

        # 4. Format Submission
        # We need 'contact_id' and 'contact'
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": preds}
        )

        # 5. Save
        sub_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}. Rows: {len(submission)}")

    def run_training_pipeline(self, sample_size=None):
        """
        Executes the full pipeline.
        """
        self.train_scouts(sample_size=sample_size)
        self.train_experts(
            sample_size=sample_size
        )  # Calls mine_hard_negatives internally
        self.optimize_threshold(sample_size=sample_size)
        self.generate_submission()
