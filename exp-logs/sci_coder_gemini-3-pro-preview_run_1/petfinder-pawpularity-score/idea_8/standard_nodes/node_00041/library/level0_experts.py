import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import save_array, load_array, print_metric, seed_everything
from library.feature_processing import get_expert_data


class Level0Trainer:
    """
    Manages the training, cross-validation, and inference of Level-0 regression experts.
    Implements 5-Fold CV on the full dataset (Train + Val) to generate OOF predictions
    and averaged test predictions.
    """

    def __init__(self, backbone_name, expert_name):
        """
        Args:
            backbone_name (str): Name of the backbone model (e.g., Config.MODEL_CLIP).
            expert_name (str): The type of expert estimator ('ridge', 'svr', 'knn', 'extratrees').
        """
        self.backbone_name = backbone_name
        self.expert_name = expert_name

        # Map the expert type to the preprocessing group defined in feature_processing.py
        if expert_name in ["ridge", "svr"]:
            self.expert_group = "linear"
        elif expert_name == "extratrees":
            self.expert_group = "tree"
        else:
            raise ValueError(f"Unknown expert type: {expert_name}")

    def get_estimator(self):
        """
        Factory method to instantiate the configured sklearn estimator.
        """
        seed_everything(Config.SEED)

        if self.expert_name == "ridge":
            # RidgeCV performs efficient Leave-One-Out CV to select alpha
            return RidgeCV(
                alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error"
            )

        elif self.expert_name == "svr":
            return SVR(
                C=Config.SVR_C,
                kernel=Config.SVR_KERNEL,
                epsilon=Config.SVR_EPSILON,
            )

        elif self.expert_name == "extratrees":
            return ExtraTreesRegressor(
                n_estimators=Config.ET_N_ESTIMATORS,
                max_depth=Config.ET_MAX_DEPTH,
                min_samples_split=Config.ET_MIN_SAMPLES_SPLIT,
                min_samples_leaf=Config.ET_MIN_SAMPLES_LEAF,
                random_state=Config.SEED,
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown expert type: {self.expert_name}")

    def run(self, load_cached_data=True):
        """
        Executes the training pipeline:
        1. Loads processed features.
        2. Merges Train and Val sets for full-dataset CV.
        3. Performs 5-Fold CV to generate OOF predictions.
        4. Predicts on Test set (averaged across folds).
        5. Caches results.

        Args:
            load_cached_data (bool): Whether to attempt loading results from cache.

        Returns:
            dict: Dictionary containing 'oof', 'test_pred', 'test_ids', 'train_targets'.
        """
        # Construct unique cache file paths
        sanitized_backbone = self.backbone_name.replace("/", "_")
        prefix = f"{sanitized_backbone}_{self.expert_name}"
        cache_dir = Config.WORKING_DIR

        path_oof = os.path.join(cache_dir, f"{prefix}_oof.npy")
        path_test_pred = os.path.join(cache_dir, f"{prefix}_test_pred.npy")
        path_test_ids = os.path.join(cache_dir, f"{prefix}_test_ids.npy")
        path_targets = os.path.join(cache_dir, f"{prefix}_train_targets.npy")

        # 1. Check Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_oof)
                and os.path.exists(path_test_pred)
                and os.path.exists(path_test_ids)
                and os.path.exists(path_targets)
            )
            if files_exist:
                print(
                    f"Loading cached Level-0 predictions for {self.backbone_name} ({self.expert_name})..."
                )
                return {
                    "oof": load_array(path_oof),
                    "test_pred": load_array(path_test_pred),
                    "test_ids": load_array(path_test_ids),
                    "train_targets": load_array(path_targets),
                }

        print(f"Training Level-0 Expert: {self.backbone_name} ({self.expert_name})...")

        # 2. Load Data
        # We use load_cached_data=True here to leverage feature_processing cache
        data = get_expert_data(
            self.backbone_name, self.expert_group, load_cached_data=True
        )

        # Merge Train and Val sets for full development set
        X_dev = np.concatenate([data["X_train"], data["X_val"]], axis=0)
        y_dev = np.concatenate([data["y_train"], data["y_val"]], axis=0)
        X_test = data["X_test"]
        ids_test = data["ids_test"]

        # 3. Setup Cross-Validation
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        oof_preds = np.zeros(len(y_dev), dtype=np.float32)
        test_preds_accum = np.zeros(len(X_test), dtype=np.float32)

        # 4. Training Loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev, y_dev)):
            X_fold_train, y_fold_train = X_dev[train_idx], y_dev[train_idx]
            X_fold_val, y_fold_val = X_dev[val_idx], y_dev[val_idx]

            # Instantiate and Fit
            model = self.get_estimator()
            model.fit(X_fold_train, y_fold_train)

            # Predict OOF
            val_pred = model.predict(X_fold_val)
            oof_preds[val_idx] = val_pred.astype(np.float32)

            # Predict Test
            test_pred = model.predict(X_test)
            test_preds_accum += test_pred.astype(np.float32)

            # Log Fold Metric
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            print(
                f"  [{self.backbone_name}][{self.expert_name}] Fold {fold+1} RMSE: {fold_rmse}"
            )

        # 5. Finalize Predictions
        avg_test_preds = test_preds_accum / Config.N_FOLDS
        total_rmse = np.sqrt(mean_squared_error(y_dev, oof_preds))
        print_metric(
            f"Overall RMSE ({self.backbone_name} - {self.expert_name})", total_rmse
        )

        # 6. Save to Cache
        save_array(path_oof, oof_preds)
        save_array(path_test_pred, avg_test_preds)
        save_array(path_test_ids, ids_test)
        save_array(path_targets, y_dev)

        return {
            "oof": oof_preds,
            "test_pred": avg_test_preds,
            "test_ids": ids_test,
            "train_targets": y_dev,
        }
