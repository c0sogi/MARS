import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.expert_models import ExpertFactory, MetaLearner
from library.data_loader import load_metadata


class StackingEngine:
    """
    Orchestrates the Multi-Scale Tri-Paradigm Stacking Ensemble.
    Manages data loading, Level-0 expert training (with CV), Level-1 meta-learning,
    and submission generation.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH
        seed_everything(Config.SEED)

    def _get_feature_path(self, mode, backbone, view):
        return os.path.join(self.working_dir, f"{mode}_{backbone}_{view}_features.npy")

    def _get_meta_path(self, mode):
        return os.path.join(self.working_dir, f"{mode}_meta.npy")

    def _get_target_path(self, mode):
        return os.path.join(self.working_dir, f"{mode}_targets.npy")

    def _get_ids_path(self, mode):
        return os.path.join(self.working_dir, f"{mode}_ids.npy")

    def load_data(self, mode):
        """
        Loads all feature sets, metadata, and targets for a given mode.
        Returns a dictionary of features keyed by (backbone, view), plus meta and targets.
        """
        data_dict = {}

        # Load features for all backbones and views
        for backbone in Config.BACKBONES.keys():
            for view in ["global", "local"]:
                path = self._get_feature_path(mode, backbone, view)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Feature file not found: {path}. Run feature extraction first."
                    )
                data_dict[(backbone, view)] = np.load(path)

        # Load metadata
        meta_path = self._get_meta_path(mode)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        metadata = np.load(meta_path)

        # Load targets
        target_path = self._get_target_path(mode)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
        targets = np.load(target_path)

        # Load IDs
        ids_path = self._get_ids_path(mode)
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"IDs file not found: {ids_path}")
        ids = np.load(ids_path)

        return data_dict, metadata, targets, ids

    def _construct_input(self, embeddings, metadata):
        """
        Concatenates image embeddings with binary metadata.
        """
        return np.hstack([embeddings, metadata])

    def train_level0(self, load_cached_data=True):
        """
        Trains Level-0 experts using 5-Fold CV.
        Generates OOF predictions (for training Meta-Learner) and Test predictions.

        Args:
            load_cached_data (bool): If True, attempts to load OOF/Test matrices from disk.

        Returns:
            tuple: (oof_matrix, train_targets, test_matrix, test_ids)
        """
        cache_oof_path = os.path.join(self.working_dir, "level0_oof.npy")
        cache_test_pred_path = os.path.join(self.working_dir, "level0_test.npy")
        cache_targets_path = os.path.join(self.working_dir, "level0_train_targets.npy")
        cache_test_ids_path = os.path.join(self.working_dir, "level0_test_ids.npy")

        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(cache_oof_path)
                and os.path.exists(cache_test_pred_path)
                and os.path.exists(cache_targets_path)
                and os.path.exists(cache_test_ids_path)
            ):
                # Validate cache dimension
                try:
                    cached_targets = np.load(cache_targets_path, mmap_mode="r")
                    current_df = load_metadata("train_all")
                    if len(cached_targets) != len(current_df):
                        print(
                            f"Level-0 Cache dimension mismatch (Cached: {len(cached_targets)}, Expected: {len(current_df)}). Retraining..."
                        )
                    else:
                        print("Loading cached Level-0 predictions...")
                        return (
                            np.load(cache_oof_path),
                            np.load(cache_targets_path),
                            np.load(cache_test_pred_path),
                            np.load(cache_test_ids_path),
                        )
                except Exception as e:
                    print(f"Error validating Level-0 cache: {e}. Retraining...")

        print("Starting Level-0 Expert Training...")

        # 2. Load Raw Data
        train_feats, train_meta, train_targets, _ = self.load_data("train_all")
        test_feats, test_meta, _, test_ids = self.load_data("test")

        # 3. Setup Storage
        # 18 Experts: 3 Backbones * 2 Views * 3 Algos
        expert_configs = []
        algorithms = ["ridge", "svr", "extratrees"]

        for backbone in Config.BACKBONES.keys():
            for view in ["global", "local"]:
                for algo in algorithms:
                    expert_configs.append((backbone, view, algo))

        n_samples = len(train_targets)
        n_experts = len(expert_configs)
        n_test = len(test_ids)

        oof_matrix = np.zeros((n_samples, n_experts), dtype=np.float32)
        test_matrix = np.zeros((n_test, n_experts), dtype=np.float32)

        # 4. Cross-Validation Loop
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        # Iterate through each expert configuration
        for col_idx, (backbone, view, algo) in enumerate(expert_configs):
            print(f"Training Expert {col_idx+1}/{n_experts}: {backbone}-{view}-{algo}")

            # Prepare Full Data for this stream
            X_train_full = self._construct_input(
                train_feats[(backbone, view)], train_meta
            )
            X_test_full = self._construct_input(test_feats[(backbone, view)], test_meta)
            input_dim = X_train_full.shape[1]

            # Temporary storage for test predictions across folds (to be averaged)
            temp_test_preds = np.zeros(n_test, dtype=np.float32)

            for fold, (train_idx, val_idx) in enumerate(
                kf.split(X_train_full, train_targets)
            ):
                # Split Data
                X_tr, y_tr = X_train_full[train_idx], train_targets[train_idx]
                X_val = X_train_full[val_idx]

                # Get Expert Model
                model = ExpertFactory.get_level0_expert(
                    algo, input_dim, n_samples=len(X_tr)
                )

                # Train
                model.fit(X_tr, y_tr)

                # Predict OOF
                val_preds = model.predict(X_val)
                oof_matrix[val_idx, col_idx] = val_preds

                # Predict Test (accumulate)
                temp_test_preds += model.predict(X_test_full)

            # Average test predictions across folds
            test_matrix[:, col_idx] = temp_test_preds / Config.N_FOLDS

            # Print quick metric for this expert
            expert_rmse = compute_rmse(train_targets, oof_matrix[:, col_idx])
            print(f"  -> Expert OOF RMSE: {expert_rmse:.5f}")

        # 5. Save to Cache
        print("Saving Level-0 predictions to cache...")
        np.save(cache_oof_path, oof_matrix)
        np.save(cache_test_pred_path, test_matrix)
        np.save(cache_targets_path, train_targets)
        np.save(cache_test_ids_path, test_ids)

        return oof_matrix, train_targets, test_matrix, test_ids

    def train_level1(self, oof_matrix, targets):
        """
        Trains the Level-1 Meta-Learner on OOF predictions.
        Calculates validation RMSE.
        """
        print("Training Level-1 Meta-Learner...")

        meta_learner = MetaLearner()
        meta_learner.fit(oof_matrix, targets)

        # Evaluate on OOF data (Proxy for CV score)
        oof_preds = meta_learner.predict(oof_matrix)
        final_rmse = compute_rmse(targets, oof_preds)

        print(f"Final Ensemble OOF RMSE: {final_rmse:.10f}")

        return meta_learner

    def generate_submission(self, meta_learner, test_matrix, test_ids):
        """
        Generates final predictions and saves submission file.
        """
        print("Generating submission...")

        final_preds = meta_learner.predict(test_matrix)

        # Clip predictions to valid range [1, 100] as per dataset description
        # Though Pawpularity is 1-100, regression might overshoot slightly
        final_preds = np.clip(final_preds, 1.0, 100.0)

        submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": final_preds})

        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        print(submission_df.head())

    def run(self, load_cached_data=True):
        """
        Main execution method.
        """
        # 1. Level-0 Training (or Load Cache)
        oof_matrix, train_targets, test_matrix, test_ids = self.train_level0(
            load_cached_data=load_cached_data
        )

        # 2. Level-1 Training
        meta_learner = self.train_level1(oof_matrix, train_targets)

        # 3. Submission
        self.generate_submission(meta_learner, test_matrix, test_ids)
