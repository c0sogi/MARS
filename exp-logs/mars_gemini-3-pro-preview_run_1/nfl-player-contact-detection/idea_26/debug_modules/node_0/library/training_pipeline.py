import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.utils import shuffle

import library.config as config
import library.utils as utils
from library.physics_engine import FeatureManager
from library.model_factory import TriEnsemble


class TrainingPipeline:
    """
    Orchestrates the VASM-E (Vector-Aligned Soft-Mining Ensemble) training strategy.
    Implements the Tri-Scout Anchored Mining Curriculum.
    """

    def __init__(self):
        self.feature_manager = FeatureManager()
        self.scouts = None
        self.experts = None
        self.best_threshold = 0.5

    def _get_hard_negative_cache_path(self):
        return os.path.join(config.CACHE_DIR, "hard_negative_indices.npy")

    def _train_scouts(self, df_train):
        """
        Phase 1: Train Scouts on a balanced subset of the gated survivors.
        """
        print("\n--- Phase 1: Tri-Scout Training ---")

        # 1. Separate Positives and Negatives
        df_pos = df_train[df_train["contact"] == 1]
        df_neg = df_train[df_train["contact"] == 0]

        # 2. Balanced Sampling (1:1 Ratio for Scouts)
        n_pos = len(df_pos)
        if len(df_neg) > n_pos:
            df_neg_sampled = df_neg.sample(n=n_pos, random_state=config.SEED)
        else:
            df_neg_sampled = df_neg

        df_scout = pd.concat([df_pos, df_neg_sampled], axis=0)
        df_scout = shuffle(df_scout, random_state=config.SEED).reset_index(drop=True)

        print(
            f"Scout Training Data: {len(df_scout)} rows (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
        )

        # 3. Prepare Features
        X_scout = df_scout[config.MODEL_FEATURES]
        y_scout = df_scout["contact"]

        # 4. Train Scouts
        self.scouts = TriEnsemble()
        # Note: We don't use validation set for scouts to save time/complexity,
        # as they are just for mining. We rely on conservative hyperparameters.
        self.scouts.fit(X_scout, y_scout)

        print("Scout models trained successfully.")

    def _mine_hard_negatives(self, df_train, load_cached_mining=True):
        """
        Phase 2: Diversity Mining.
        Run Scouts on the full gated training set to find hard negatives.
        Hard Negative: Any instance where ANY scout predicts > HARD_NEGATIVE_THRESHOLD.
        """
        print("\n--- Phase 2: Diversity Mining ---")

        cache_path = self._get_hard_negative_cache_path()

        # Check cache
        if load_cached_mining and os.path.exists(cache_path):
            print(f"Loading cached hard negative indices from {cache_path}...")
            return np.load(cache_path)

        print(f"Mining hard negatives from {len(df_train)} samples...")

        X_full = df_train[config.MODEL_FEATURES]

        # Get predictions from individual scouts
        # TriEnsemble stores models in self.models dict
        hard_negative_mask = np.zeros(len(df_train), dtype=bool)

        for name, model in self.scouts.models.items():
            print(f"Scoring with Scout {name}...")
            # Handle different predict_proba signatures if necessary, but sklearn/lgb/xgb are consistent here
            probs = model.predict_proba(X_full)[:, 1]

            # Update mask: Union of hard negatives
            mask = probs > config.HARD_NEGATIVE_THRESHOLD
            hard_negative_mask = hard_negative_mask | mask

            count = np.sum(mask)
            print(f"  > Found {count} candidates with {name}")

        # Filter to ensure we only select actual negatives (contact == 0)
        # (Though technically high prob positives are just correctly classified,
        # we want to mine the *negatives* that look like positives)
        is_negative = (df_train["contact"] == 0).values
        final_mask = hard_negative_mask & is_negative

        hard_negative_indices = df_train.index[final_mask].to_numpy()

        print(f"Total Unique Hard Negatives Mined: {len(hard_negative_indices)}")

        # Cache results
        np.save(cache_path, hard_negative_indices)

        return hard_negative_indices

    def _train_experts(self, df_train, hard_negative_indices, df_val):
        """
        Phase 3: Anchored Expert Training with Soft Targets.
        Dataset: All Positives + Hard Negatives + Random Anchors.
        """
        print("\n--- Phase 3: Anchored Expert Training ---")

        # 1. Construct Anchored Dataset
        # Indices
        pos_indices = df_train[df_train["contact"] == 1].index.to_numpy()

        # Random Anchors (Easy Negatives)
        # Candidates are negatives that are NOT in hard_negative_indices
        all_neg_indices = df_train[df_train["contact"] == 0].index.to_numpy()
        # Set difference to find easy negatives
        # Note: Using np.setdiff1d can be slow on large arrays.
        # Faster approach: boolean mask
        is_hard = np.zeros(len(df_train), dtype=bool)
        is_hard[hard_negative_indices] = True

        # Easy negatives are Negatives AND NOT Hard
        is_negative = df_train["contact"] == 0
        is_easy = is_negative & (~is_hard)
        easy_neg_indices = df_train.index[is_easy].to_numpy()

        # Determine number of anchors
        # Ratio relative to (Positives + Hard Negatives) or just Positives?
        # Config says "Ratio of Anchors to Positives/Hard Negatives in the Expert Set"
        # Let's target a size proportional to the "difficult" part of the dataset
        n_difficult = len(pos_indices) + len(hard_negative_indices)
        n_anchors = int(n_difficult * config.ANCHOR_RATIO)

        # Sample anchors
        if len(easy_neg_indices) > n_anchors:
            rng = np.random.RandomState(config.SEED)
            anchor_indices = rng.choice(easy_neg_indices, size=n_anchors, replace=False)
        else:
            anchor_indices = easy_neg_indices

        print(f"Constructing Expert Dataset:")
        print(f"  Positives: {len(pos_indices)}")
        print(f"  Hard Negatives: {len(hard_negative_indices)}")
        print(f"  Anchors (Easy Neg): {len(anchor_indices)}")

        final_indices = np.concatenate(
            [pos_indices, hard_negative_indices, anchor_indices]
        )
        df_expert = df_train.loc[final_indices].copy()

        # Shuffle
        df_expert = shuffle(df_expert, random_state=config.SEED).reset_index(drop=True)

        # 2. Apply Temporal Label Smoothing (Soft Targets)
        # We must apply this BEFORE selecting X and y.
        # Note: gaussian_smooth_labels requires sorted data by play/step,
        # but we just shuffled. However, the function sorts internally.
        # IMPORTANT: Smoothing should ideally be done on the full continuous series
        # before subsampling, or we lose temporal context.
        # BUT: We are working with a subsampled dataset now.
        # Strategy: Apply smoothing to the FULL df_train first, then subset.
        # To save memory, let's do it on df_train before subsetting?
        # No, df_train is huge.
        # Let's apply smoothing to the FULL df_train in a memory efficient way
        # or assume the subsetting breaks temporal continuity anyway?
        # If we subset, we lose neighbors.
        # CORRECT APPROACH: Apply smoothing to the FULL df_train at the start of this method,
        # then extract the rows.

        print("Applying Temporal Label Smoothing to full training set (for context)...")
        # We operate on a view or copy? df_train is passed by reference.
        # Let's modify df_train in place or return new. The util returns new/modified.
        # To avoid modifying the original passed df_train repeatedly if called multiple times,
        # check if column exists.
        if "contact_smooth" not in df_train.columns:
            df_train = utils.gaussian_smooth_labels(
                df_train, sigma=config.SMOOTHING_SIGMA
            )

        # Re-extract df_expert from the now-smoothed df_train
        df_expert = df_train.loc[final_indices].copy()
        df_expert = shuffle(df_expert, random_state=config.SEED).reset_index(drop=True)

        # 3. Prepare Training Data
        X_train = df_expert[config.MODEL_FEATURES]
        # Use Soft Targets
        y_train = df_expert["contact_smooth"]

        # 4. Prepare Validation Data
        # Validation should use binary targets for metric calculation,
        # but soft targets for loss? Usually val metric is hard (AP/AUC).
        # We pass binary targets to the fit method's eval_set for metric calculation.
        X_val = df_val[config.MODEL_FEATURES]
        y_val = df_val["contact"]

        # 5. Train Experts
        self.experts = TriEnsemble()
        self.experts.fit(X_train, y_train, X_val, y_val)

        print("Expert models trained successfully.")

    def run(self, debug_sample=None, load_cached_data=True):
        """
        Main execution method.
        """
        print(f"Starting VASM-E Pipeline (Exp: {config.EXP_NAME})")

        # 1. Load and Process Data
        # Train
        df_train = self.feature_manager.process_data(
            split="train", load_cached_data=load_cached_data, debug_sample=debug_sample
        )
        # Val
        df_val = self.feature_manager.process_data(
            split="val", load_cached_data=load_cached_data, debug_sample=debug_sample
        )

        # 2. Phase 1: Train Scouts
        # We only train scouts if we don't have cached mining results
        # OR if we want to force retrain.
        # For this pipeline, we check if mining cache exists.
        # If mining cache exists, we can skip scout training to save time.
        mining_cache_path = self._get_hard_negative_cache_path()
        run_scouts = not (load_cached_data and os.path.exists(mining_cache_path))

        if run_scouts:
            self._train_scouts(df_train)
            hard_neg_indices = self._mine_hard_negatives(
                df_train, load_cached_mining=False
            )
            # Free memory
            self.scouts = None
            gc.collect()
        else:
            print("Skipping Scout Training (Loading cached hard negatives)...")
            hard_neg_indices = self._mine_hard_negatives(
                df_train, load_cached_mining=True
            )

        # 3. Phase 3: Train Experts
        self._train_experts(df_train, hard_neg_indices, df_val)

        # 4. Evaluation & Threshold Optimization
        print("\n--- Evaluation ---")
        val_probs = self.experts.predict_proba(df_val[config.MODEL_FEATURES])
        y_val = df_val["contact"].values

        best_thresh, best_mcc = utils.find_best_threshold(y_val, val_probs)
        print(f"Best Threshold: {best_thresh:.4f}")
        print(f"Validation MCC: {best_mcc:.8f}")

        # Save Threshold
        self.best_threshold = best_thresh
        np.save(
            os.path.join(config.MODEL_DIR, "best_threshold.npy"), self.best_threshold
        )

        # Save Model
        self.experts.save("expert_ensemble.joblib")

        # Clean up large dataframes
        del df_train, df_val, hard_neg_indices
        gc.collect()

        print("Pipeline execution completed.")
        return best_mcc
