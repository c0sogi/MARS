import os
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.features import FeatureEngineer
from library.models import TriEnsemble


class Trainer:
    """
    Orchestrates the QGVSA-E training pipeline:
    1. Scout Training (Balanced)
    2. Hard Negative Mining (Full Gated)
    3. Expert Training (Anchored + Smoothed Labels)
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def load_and_gate_data(self, dataset_type="train", load_cached_data=True):
        """
        Loads features and applies Quadratic Reachability Gating.
        Returns both the gated subset and the full dataset.
        """
        # Load full features
        df = self.feature_engineer.process_data(
            dataset_type, load_cached_data=load_cached_data
        )

        # Apply Gating: Keep if quadratic_min_dist < Threshold
        # Sentinel values for Ground are handled in features.py (usually 0 or -1, ensuring they pass if relevant)
        mask = df["quadratic_min_dist"] < Config.GATING_THRESHOLD
        df_gated = df[mask].copy().reset_index(drop=True)

        print(
            f"[{dataset_type.upper()}] Gating applied: {len(df)} -> {len(df_gated)} rows ({(len(df_gated)/len(df))*100:.2f}% retained)."
        )
        return df_gated, df

    def train_scouts(self, df_train_gated, df_val_gated):
        """
        Phase 1: Train Scouts on a balanced subset of gated survivors.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # 1. Create Balanced Dataset
        pos_mask = df_train_gated["contact"] == 1
        neg_mask = df_train_gated["contact"] == 0

        df_pos = df_train_gated[pos_mask]
        df_neg = df_train_gated[neg_mask]

        # Sample negatives to match positives
        n_pos = len(df_pos)
        if len(df_neg) > n_pos:
            df_neg_sampled = df_neg.sample(n=n_pos, random_state=Config.SEED)
        else:
            df_neg_sampled = df_neg

        df_balanced = (
            pd.concat([df_pos, df_neg_sampled])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        print(f"Scout Training Data: {len(df_balanced)} rows (Balanced).")

        # Prepare X, y
        X_train = df_balanced[Config.FEATURES]
        y_train = df_balanced["contact"]

        X_val = df_val_gated[Config.FEATURES]
        y_val = df_val_gated["contact"]

        # Train Ensemble
        scout_ensemble = TriEnsemble()
        scout_ensemble.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Save
        scout_ensemble.save_models(prefix="scout")
        return scout_ensemble

    def mine_hard_negatives(
        self, df_train_gated, scout_ensemble, load_cached_data=True
    ):
        """
        Phase 2: Mine Hard Negatives using Scouts.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")

        cache_path = Config.CACHE_HARD_NEGATIVES

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading hard negative indices from {cache_path}")
            return np.load(cache_path)

        # Identify Negatives in Gated Set
        neg_indices = df_train_gated.index[df_train_gated["contact"] == 0].tolist()

        # Predict on all negatives
        X_neg = df_train_gated.loc[neg_indices, Config.FEATURES]

        print("Predicting with Scout LGBM...")
        p_lgbm = scout_ensemble.lgbm.predict_proba(X_neg)[:, 1]

        print("Predicting with Scout XGB...")
        p_xgb = scout_ensemble.xgb.predict_proba(X_neg)[:, 1]

        print("Predicting with Scout CatBoost...")
        p_cat = scout_ensemble.cat.predict_proba(X_neg)[:, 1]

        # Compute Union Condition: Any model > Threshold
        max_proba = np.maximum(np.maximum(p_lgbm, p_xgb), p_cat)

        hard_mask = max_proba > Config.HARD_NEGATIVE_THRESHOLD
        hard_indices = np.array(neg_indices)[hard_mask]

        print(
            f"Mined {len(hard_indices)} Hard Negatives out of {len(neg_indices)} negatives."
        )

        # Save
        np.save(cache_path, hard_indices)

        return hard_indices

    def smooth_labels(self, df):
        """
        Applies temporal Gaussian smoothing to labels.
        """
        print("Applying Temporal Label Smoothing...")

        # Sort for temporal consistency
        # Note: We rely on the index alignment of pandas to map back to original df
        df_sorted = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        ).copy()

        def smooth_func(x):
            if len(x) < 2:
                return x.astype(float)
            return gaussian_filter1d(
                x.astype(float), sigma=Config.LABEL_SMOOTHING_SIGMA, mode="nearest"
            )

        # Apply smoothing per group
        df_sorted["contact_smooth"] = df_sorted.groupby(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        )["contact"].transform(smooth_func)

        # Map back to original dataframe via index alignment
        df["contact_smooth"] = df_sorted["contact_smooth"]

        return df

    def create_anchored_dataset(self, df_train_gated, hard_indices):
        """
        Phase 3 Setup: Construct Expert Dataset (Positives + Hard Negatives + Random Anchors).
        """
        print("Constructing Anchored Dataset...")

        # 1. Positives
        df_pos = df_train_gated[df_train_gated["contact"] == 1]

        # 2. Hard Negatives
        df_hard = df_train_gated.loc[hard_indices]

        # 3. Easy Negatives (Anchors)
        # All negatives NOT in hard_indices
        all_neg_indices = set(df_train_gated.index[df_train_gated["contact"] == 0])
        hard_indices_set = set(hard_indices)
        easy_indices = list(all_neg_indices - hard_indices_set)

        # Calculate number of anchors
        n_anchors = int((len(df_pos) + len(df_hard)) * Config.ANCHOR_RATIO)

        # Sample anchors
        if len(easy_indices) > n_anchors:
            rng = np.random.RandomState(Config.SEED)
            anchor_indices = rng.choice(easy_indices, size=n_anchors, replace=False)
        else:
            anchor_indices = easy_indices

        df_anchors = df_train_gated.loc[anchor_indices]

        # Combine
        df_expert = (
            pd.concat([df_pos, df_hard, df_anchors])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        print(
            f"Expert Dataset: {len(df_expert)} rows (Pos: {len(df_pos)}, Hard: {len(df_hard)}, Anchors: {len(df_anchors)})"
        )

        return df_expert

    def train_experts(self, df_expert, df_val_gated):
        """
        Phase 3: Train Expert Ensemble using smoothed labels.
        """
        print("\n--- Phase 3: Training Experts ---")

        # Use smoothed target if available
        target_col = (
            "contact_smooth" if "contact_smooth" in df_expert.columns else "contact"
        )
        print(f"Using target column: {target_col}")

        X_train = df_expert[Config.FEATURES]
        y_train = df_expert[target_col]

        # Validation always uses binary ground truth
        X_val = df_val_gated[Config.FEATURES]
        y_val = df_val_gated["contact"]

        expert_ensemble = TriEnsemble()
        expert_ensemble.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        expert_ensemble.save_models(prefix="expert")
        return expert_ensemble

    def find_best_threshold(self, ensemble, df_val_full):
        """
        Optimizes decision threshold on the full validation set (including gated-out rows).
        """
        print("\n--- Optimizing Threshold ---")

        # Identify Gated Rows for Prediction
        mask = df_val_full["quadratic_min_dist"] < Config.GATING_THRESHOLD

        # Predict only on gated rows
        X_gated = df_val_full.loc[mask, Config.FEATURES]
        y_probs_gated = ensemble.predict_proba(X_gated)[:, 1]

        # Construct full probability vector (non-gated = 0.0)
        y_probs_full = np.zeros(len(df_val_full))
        y_probs_full[mask] = y_probs_gated

        y_true_full = df_val_full["contact"].values

        # Threshold Search
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_mcc = -1
        best_th = 0.5

        for th in thresholds:
            y_pred = (y_probs_full >= th).astype(int)
            mcc = matthews_corrcoef(y_true_full, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        print(f"Best Threshold: {best_th:.16f} with MCC: {best_mcc:.16f}")

        # Save threshold
        np.save(
            os.path.join(Config.CACHE_MODELS, "best_threshold.npy"), np.array([best_th])
        )

        return best_th

    def generate_submission(self, ensemble, threshold):
        """
        Generates submission.csv for the test set.
        """
        print("\n--- Generating Submission ---")

        # Load Test Data
        df_test_gated, df_test_full = self.load_and_gate_data("test")

        # Predict on Gated Test Rows
        X_test_gated = df_test_gated[Config.FEATURES]
        probs_gated = ensemble.predict_proba(X_test_gated)[:, 1]

        # Map predictions back to full test set
        # We need to align with df_test_full
        # df_test_gated is a subset. We can use the mask logic again.
        mask = df_test_full["quadratic_min_dist"] < Config.GATING_THRESHOLD

        full_probs = np.zeros(len(df_test_full))
        full_probs[mask] = probs_gated

        # Apply Threshold
        predictions = (full_probs >= threshold).astype(int)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"contact_id": df_test_full["contact_id"], "contact": predictions}
        )

        # Save
        sub_df.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        print(
            f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH} with {len(sub_df)} rows."
        )

    def run(self):
        """
        Executes the full training pipeline.
        """
        # 1. Load Data
        df_train_gated, _ = self.load_and_gate_data("train")
        df_val_gated, df_val_full = self.load_and_gate_data("val")

        # 2. Train Scouts
        scout_ensemble = self.train_scouts(df_train_gated, df_val_gated)

        # 3. Mine Hard Negatives
        hard_indices = self.mine_hard_negatives(df_train_gated, scout_ensemble)

        # 4. Smooth Labels (on full gated train to preserve context where possible)
        df_train_smoothed = self.smooth_labels(df_train_gated)

        # 5. Create Anchored Dataset
        df_expert_train = self.create_anchored_dataset(df_train_smoothed, hard_indices)

        # 6. Train Experts
        expert_ensemble = self.train_experts(df_expert_train, df_val_gated)

        # 7. Evaluate
        best_threshold = self.find_best_threshold(expert_ensemble, df_val_full)

        # 8. Generate Submission
        self.generate_submission(expert_ensemble, best_threshold)
