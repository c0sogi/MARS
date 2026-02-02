import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Optional, List

from library.config import Config
from library.feature_extraction import extract_and_cache_features
from library.stacking_models import Level0Expert, Level1MetaLearner
from library.utils import seed_everything, compute_rmse


class StackingManager:
    """
    Orchestrates the Multi-Paradigm Stacking Ensemble pipeline.
    Manages Level 0 experts (feature extraction + RidgeCV) and Level 1 Meta-Learner.
    """

    def __init__(
        self,
        subset_size: Optional[int] = None,
        load_cached_data: bool = True,
        n_folds: int = Config.N_FOLDS,
        working_dir: str = Config.WORKING_DIR,
    ):
        """
        Args:
            subset_size (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Whether to load features from cache.
            n_folds (int): Number of CV folds for Level 0.
            working_dir (str): Directory to store intermediate files.
        """
        self.subset_size = subset_size
        self.load_cached_data = load_cached_data
        self.n_folds = n_folds
        self.working_dir = working_dir

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        self.model_keys = list(Config.MODEL_CONFIGS.keys())
        seed_everything(Config.SEED)

        # Containers for Level 0 outputs
        # Keys: model_key, Values: np.ndarray
        self.L0_oof_preds = {}  # Train OOF predictions
        self.L0_val_preds = {}  # Validation predictions (averaged over folds)
        self.L0_test_preds = {}  # Test predictions (averaged over folds)

        # Ground truths / IDs
        self.train_targets = None
        self.val_targets = None
        self.test_ids = None

        self.meta_learner = None

    def train_level_0(self):
        """
        Trains Level 0 experts using K-Fold CV.
        Generates OOF predictions for Train, and averaged predictions for Val and Test.
        """
        print("=== Starting Level 0 Training ===")

        for model_key in self.model_keys:
            print(f"\nProcessing Expert: {model_key}")

            # 1. Load Data for all splits
            train_data = extract_and_cache_features(
                model_key, "train", self.load_cached_data, subset_size=self.subset_size
            )
            val_data = extract_and_cache_features(
                model_key, "val", self.load_cached_data, subset_size=self.subset_size
            )
            test_data = extract_and_cache_features(
                model_key, "test", self.load_cached_data, subset_size=self.subset_size
            )

            # Store targets/IDs if not already stored
            if self.train_targets is None:
                self.train_targets = train_data["targets"]
            if self.val_targets is None:
                self.val_targets = val_data["targets"]
            if self.test_ids is None:
                self.test_ids = test_data["ids"]

            # Unpack features
            X_train_img = train_data["features"]
            X_train_meta = train_data["meta"]
            y_train = train_data["targets"]

            X_val_img = val_data["features"]
            X_val_meta = val_data["meta"]

            X_test_img = test_data["features"]
            X_test_meta = test_data["meta"]

            # 2. Prepare arrays for predictions
            oof_preds = np.zeros(len(y_train))
            val_preds_accum = np.zeros(len(X_val_img))
            test_preds_accum = np.zeros(len(X_test_img))

            # 3. K-Fold CV
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=Config.SEED)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_img)):
                # Split data
                X_tr_img, X_va_img = X_train_img[train_idx], X_train_img[valid_idx]
                X_tr_meta, X_va_meta = X_train_meta[train_idx], X_train_meta[valid_idx]
                y_tr, y_va = y_train[train_idx], y_train[valid_idx]

                # Initialize and Train Expert
                expert = Level0Expert()
                expert.fit(X_tr_img, X_tr_meta, y_tr)

                # Predict OOF (Validation fold)
                pred_oof = expert.predict(X_va_img, X_va_meta)
                oof_preds[valid_idx] = pred_oof

                # Predict on fixed Validation Set
                pred_val = expert.predict(X_val_img, X_val_meta)
                val_preds_accum += pred_val

                # Predict on Test Set
                pred_test = expert.predict(X_test_img, X_test_meta)
                test_preds_accum += pred_test

            # Average predictions
            val_preds_avg = val_preds_accum / self.n_folds
            test_preds_avg = test_preds_accum / self.n_folds

            # Calculate overall OOF RMSE
            # Flatten targets to ensure shape match with preds
            oof_rmse = compute_rmse(y_train.ravel(), oof_preds)
            print(f"  {model_key} OOF RMSE: {oof_rmse}")

            # Calculate RMSE on fixed Validation Set
            val_rmse = compute_rmse(self.val_targets.ravel(), val_preds_avg)
            print(f"  {model_key} Fixed Val RMSE: {val_rmse}")

            # Store results
            self.L0_oof_preds[model_key] = oof_preds
            self.L0_val_preds[model_key] = val_preds_avg
            self.L0_test_preds[model_key] = test_preds_avg

            # Cache intermediate predictions
            np.save(os.path.join(self.working_dir, f"{model_key}_oof.npy"), oof_preds)
            np.save(
                os.path.join(self.working_dir, f"{model_key}_val_pred.npy"),
                val_preds_avg,
            )
            np.save(
                os.path.join(self.working_dir, f"{model_key}_test_pred.npy"),
                test_preds_avg,
            )

    def train_level_1(self):
        """
        Trains the Level 1 Meta-Learner on the OOF predictions from Level 0.
        Evaluates on the fixed Validation set.
        """
        print("\n=== Starting Level 1 Training ===")

        if not self.L0_oof_preds:
            raise RuntimeError(
                "Level 0 predictions not found. Run train_level_0 first."
            )

        # Construct Feature Matrices for Level 1
        # Use sorted keys to ensure consistent column order
        sorted_keys = sorted(self.model_keys)

        X_train_L1 = np.column_stack([self.L0_oof_preds[k] for k in sorted_keys])
        X_val_L1 = np.column_stack([self.L0_val_preds[k] for k in sorted_keys])

        y_train = self.train_targets.ravel()
        y_val = self.val_targets.ravel()

        # Initialize and Train Meta Learner
        self.meta_learner = Level1MetaLearner(alpha=1.0, positive=True)
        self.meta_learner.fit(X_train_L1, y_train)

        # Evaluate on Train (OOF Stack)
        train_preds = self.meta_learner.predict(X_train_L1)
        train_rmse = compute_rmse(y_train, train_preds)
        print(f"  Level 1 Train (OOF Stack) RMSE: {train_rmse}")

        # Evaluate on Validation
        val_preds = self.meta_learner.predict(X_val_L1)
        val_rmse = compute_rmse(y_val, val_preds)
        print(f"  Level 1 Validation RMSE: {val_rmse}")

        # Save Meta Learner
        save_path = os.path.join(self.working_dir, "meta_learner.joblib")
        self.meta_learner.save(save_path)

    def predict(self):
        """
        Generates final predictions for the test set and saves to submission.csv.
        """
        print("\n=== Generating Submission ===")

        if self.meta_learner is None:
            raise RuntimeError("Meta learner not trained.")

        sorted_keys = sorted(self.model_keys)
        X_test_L1 = np.column_stack([self.L0_test_preds[k] for k in sorted_keys])

        final_preds = self.meta_learner.predict(X_test_L1)

        # Create submission DataFrame
        submission = pd.DataFrame({"Id": self.test_ids, "Pawpularity": final_preds})

        # Save
        submission_path = Config.SUBMISSION_PATH
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(submission.head())
