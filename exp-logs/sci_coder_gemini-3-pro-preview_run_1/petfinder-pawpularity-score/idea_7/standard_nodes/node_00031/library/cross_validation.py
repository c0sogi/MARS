import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, rmse_score
from library.data_loader import get_dataloader
from library.feature_extractor import run_extraction
from library.ensemble_components import (
    get_linear_expert,
    get_kernel_expert,
    get_partitioning_expert,
    get_meta_learner,
)


class CrossValidator:
    """
    Manages the 5-Fold Cross-Validation and Stacking Ensemble workflow.
    Implements the Dual-View Tri-Paradigm Stacking strategy.
    """

    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Ensure reproducibility
        seed_everything(self.seed)

        # Define the Grid of Experts
        self.backbones = list(Config.BACKBONES.keys())  # ['clip', 'dinov2', 'convnext']
        self.views = ["warped", "preserved"]
        self.experts = {
            "ridge": get_linear_expert,
            "svr": get_kernel_expert,
            "extratrees": get_partitioning_expert,
        }

        # Generate feature column names for Level 1 to ensure consistent ordering
        self.feature_cols = []
        for bb in self.backbones:
            for view in self.views:
                for exp in self.experts.keys():
                    self.feature_cols.append(f"{bb}_{view}_{exp}")

    def _load_raw_features(self, split, load_cached_data=True):
        """
        Loads (or extracts) features for a specific split across all backbones and views.
        Ensures alignment of IDs.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.

        Returns:
            tuple: (feature_map, meta, targets, ids)
                feature_map is a dict: {backbone: {view: np.array}}
        """
        feature_map = {}
        meta_ref = None
        targets_ref = None
        ids_ref = None

        # Determine metadata path based on split
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        for bb in self.backbones:
            feature_map[bb] = {}
            for view in self.views:
                # Create DataLoader (must be deterministic order -> shuffle=False)
                loader = get_dataloader(
                    csv_path=meta_path,
                    view_mode=view,
                    backbone_type=bb,
                    shuffle=False,
                    debug=self.debug,
                )

                # Extract or Load Features
                feats, ids, meta, targets = run_extraction(
                    dataloader=loader,
                    backbone_key=bb,
                    split=split,
                    view_mode=view,
                    load_cached_data=load_cached_data,
                )

                feature_map[bb][view] = feats

                # Alignment Check
                if ids_ref is None:
                    ids_ref = ids
                    meta_ref = meta
                    targets_ref = targets
                else:
                    if not np.array_equal(ids_ref, ids):
                        raise ValueError(
                            f"ID mismatch detected in {split} split for {bb}/{view}."
                        )

        return feature_map, meta_ref, targets_ref, ids_ref

    def train_level0(self, load_cached_preds=True):
        """
        Trains Level-0 experts using 5-Fold CV.
        Generates OOF predictions for Train, and averaged predictions for Val and Test.
        Implements caching for the resulting prediction matrices.

        Args:
            load_cached_preds (bool): If True, attempts to load L0 matrices from disk.

        Returns:
            tuple: (oof_preds, train_targets, val_preds, val_targets, test_preds, test_ids)
        """
        # Define cache paths
        cache_files = {
            "oof": os.path.join(self.working_dir, "l0_oof.npy"),
            "val": os.path.join(self.working_dir, "l0_val.npy"),
            "test": os.path.join(self.working_dir, "l0_test.npy"),
            "y_train": os.path.join(self.working_dir, "l0_train_targets.npy"),
            "y_val": os.path.join(self.working_dir, "l0_val_targets.npy"),
            "id_test": os.path.join(self.working_dir, "l0_test_ids.npy"),
        }

        # 1. Attempt Cache Load
        if load_cached_preds and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading Level-0 predictions from cache...")
            return (
                np.load(cache_files["oof"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["val"]),
                np.load(cache_files["y_val"]),
                np.load(cache_files["test"]),
                np.load(cache_files["id_test"]),
            )

        print("Starting Level-0 Training (5-Fold CV)...")

        # 2. Load Raw Features for all splits
        # We use load_cached_data=True for features because extraction is expensive
        # and handled by run_extraction's own cache logic.
        train_feats, train_meta, train_targets, train_ids = self._load_raw_features(
            "train", load_cached_data=True
        )
        val_feats, val_meta, val_targets, val_ids = self._load_raw_features(
            "val", load_cached_data=True
        )
        test_feats, test_meta, test_targets, test_ids = self._load_raw_features(
            "test", load_cached_data=True
        )

        # 3. Initialize Prediction Containers
        n_train = len(train_ids)
        n_val = len(val_ids)
        n_test = len(test_ids)
        n_experts = len(self.feature_cols)

        oof_preds = np.zeros((n_train, n_experts), dtype=np.float32)
        val_preds = np.zeros((n_val, n_experts), dtype=np.float32)
        test_preds = np.zeros((n_test, n_experts), dtype=np.float32)

        # 4. Iterate through the Grid of Experts
        col_idx = 0
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for bb in self.backbones:
            for view in self.views:
                # Prepare Inputs: Concatenate Image Embeddings + Binary Metadata
                # Shape: (N, Embed_Dim + 12)
                X_train_full = np.hstack([train_feats[bb][view], train_meta])
                X_val_full = np.hstack([val_feats[bb][view], val_meta])
                X_test_full = np.hstack([test_feats[bb][view], test_meta])

                for exp_name, exp_factory in self.experts.items():
                    print(
                        f"Training Expert [{col_idx + 1}/{n_experts}]: {bb} | {view} | {exp_name}"
                    )

                    # Accumulators for averaging Test/Val predictions across folds
                    val_fold_accum = np.zeros(n_val, dtype=np.float32)
                    test_fold_accum = np.zeros(n_test, dtype=np.float32)

                    # K-Fold Loop
                    for fold, (train_idx, valid_idx) in enumerate(
                        kf.split(X_train_full, train_targets)
                    ):
                        X_tr, y_tr = X_train_full[train_idx], train_targets[train_idx]
                        X_va = X_train_full[valid_idx]

                        # Train Expert
                        model = exp_factory(random_state=self.seed)
                        model.fit(X_tr, y_tr)

                        # Predict OOF
                        oof_preds[valid_idx, col_idx] = model.predict(X_va)

                        # Predict Val/Test (Accumulate)
                        val_fold_accum += model.predict(X_val_full)
                        test_fold_accum += model.predict(X_test_full)

                    # Average predictions
                    val_preds[:, col_idx] = val_fold_accum / self.n_folds
                    test_preds[:, col_idx] = test_fold_accum / self.n_folds

                    # Log Expert Performance
                    expert_rmse = rmse_score(train_targets, oof_preds[:, col_idx])
                    print(f"  -> Expert OOF RMSE: {expert_rmse:.6f}")

                    col_idx += 1

        # 5. Save to Cache
        print("Saving Level-0 predictions to cache...")
        os.makedirs(self.working_dir, exist_ok=True)
        np.save(cache_files["oof"], oof_preds)
        np.save(cache_files["y_train"], train_targets)
        np.save(cache_files["val"], val_preds)
        np.save(cache_files["y_val"], val_targets)
        np.save(cache_files["test"], test_preds)
        np.save(cache_files["id_test"], test_ids)

        return (
            oof_preds,
            train_targets,
            val_preds,
            val_targets,
            test_preds,
            test_ids,
        )

    def train_level1(
        self, oof_preds, train_targets, val_preds, val_targets, test_preds, test_ids
    ):
        """
        Trains the Meta-Learner (Bayesian Ridge) on Level-0 OOFs.
        Generates final submission.
        """
        print("\n=== Training Level-1 Meta-Learner ===")

        meta_model = get_meta_learner(random_state=self.seed)

        # 1. Evaluate Meta-Learner using CV on Level-0 OOFs (Meta-OOF)
        # This gives a reliable estimate of the ensemble's performance on unseen data
        print("Evaluating Ensemble via Meta-CV...")
        meta_oof_preds = cross_val_predict(
            meta_model, oof_preds, train_targets, cv=self.n_folds, n_jobs=-1
        )
        cv_rmse = rmse_score(train_targets, meta_oof_preds)
        print(f"Final Ensemble CV RMSE: {cv_rmse:.10f}")

        # 2. Train on Full Level-0 OOF Data
        print("Retraining Meta-Learner on full OOF dataset...")
        meta_model.fit(oof_preds, train_targets)

        # 3. Validate on Hold-out Validation Set (External Check)
        meta_val_preds = meta_model.predict(val_preds)
        val_rmse = rmse_score(val_targets, meta_val_preds)
        print(f"Final Ensemble Hold-out Val RMSE: {val_rmse:.10f}")

        # 4. Predict on Test Set
        print("Predicting on Test Set...")
        final_test_preds = meta_model.predict(test_preds)

        # 5. Generate Submission File
        submission = pd.DataFrame({"Id": test_ids, "Pawpularity": final_test_preds})

        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")

        return cv_rmse, val_rmse

    def run(self):
        """
        Executes the full pipeline.
        """
        # Step 1: Level 0 (Base Experts)
        # We allow loading cached predictions to save time if restarting
        (
            oof,
            y_train,
            val_p,
            y_val,
            test_p,
            test_ids,
        ) = self.train_level0(load_cached_preds=True)

        # Step 2: Level 1 (Meta Learner)
        return self.train_level1(oof, y_train, val_p, y_val, test_p, test_ids)
