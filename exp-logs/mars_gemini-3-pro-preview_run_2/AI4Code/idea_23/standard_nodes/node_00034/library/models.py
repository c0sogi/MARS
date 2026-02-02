import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from library.config import Config


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor.
    Uses Ridge Regression on high-dimensional TF-IDF features.
    """

    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "stage1_ridge_model.joblib")
        self.oof_cache_path = os.path.join(self.working_dir, "stage1_oof_preds.npy")

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        self.model = Ridge(
            alpha=self.config.RIDGE_ALPHA,
            solver=self.config.RIDGE_SOLVER,
            random_state=self.config.SEED,
        )

    def get_oof_predictions(
        self, X, y: np.ndarray, groups: np.ndarray, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates Out-Of-Fold predictions using GroupKFold Cross-Validation.
        Implements strict caching logic.

        Args:
            X: Sparse TF-IDF matrix.
            y: Target array (ranks).
            groups: Group array (ancestor_ids).
            load_cached_data: Whether to try loading from cache.

        Returns:
            np.ndarray: OOF predictions aligned with X.
        """
        # 1. Try to load cached data
        if load_cached_data and os.path.exists(self.oof_cache_path):
            print(f"Loading cached OOF predictions from {self.oof_cache_path}")
            return np.load(self.oof_cache_path)

        # 2. Compute from scratch
        print("Computing Stage 1 OOF predictions...")
        oof_preds = np.zeros(len(y), dtype=np.float32)

        # 5-Fold Group CV to prevent leakage via ancestors
        gkf = GroupKFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            # Slice data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]

            # Train fold model
            model = Ridge(
                alpha=self.config.RIDGE_ALPHA,
                solver=self.config.RIDGE_SOLVER,
                random_state=self.config.SEED,
            )
            model.fit(X_train, y_train)

            # Predict
            fold_preds = model.predict(X_val)
            oof_preds[val_idx] = fold_preds

            # Optional: Print fold metric (simple MSE check)
            fold_mse = np.mean((y[val_idx] - fold_preds) ** 2)
            print(f"Stage 1 Fold {fold+1} MSE: {fold_mse}")

        # 3. Save to cache
        print(f"Saving OOF predictions to {self.oof_cache_path}")
        np.save(self.oof_cache_path, oof_preds)

        return oof_preds

    def fit(self, X, y):
        """
        Retrains the Ridge model on the full dataset for inference.
        """
        print("Training Stage 1 Ridge model on full dataset...")
        self.model.fit(X, y)

        print(f"Saving Stage 1 model to {self.model_path}")
        joblib.dump(self.model, self.model_path)
        return self

    def predict(self, X):
        """
        Predicts using the trained model. Loads from disk if not in memory.
        """
        # Check if model is fitted (Ridge sets coef_ attribute)
        if not hasattr(self.model, "coef_"):
            if os.path.exists(self.model_path):
                print(f"Loading Stage 1 model from {self.model_path}")
                self.model = joblib.load(self.model_path)
            else:
                raise RuntimeError("Stage 1 model is not fitted and no cache found.")

        return self.model.predict(X)


class Stage2LGBM:
    """
    Stage 2: Content-Aware Gradient Booster.
    Uses LightGBM to refine ranks based on Ridge predictions,
    positional anchors, and injected semantic content.
    """

    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "stage2_lgbm_model.joblib")

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with Early Stopping.

        Args:
            X_train, X_val: Feature matrices (pandas DataFrame or numpy array).
            y_train, y_val: Target arrays.
        """
        print("Training Stage 2 LightGBM model...")

        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks for logging and early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=self.config.LGBM_VERBOSE_EVAL),
        ]

        # Train
        self.model = lgb.train(
            params=self.config.LGBM_PARAMS,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        print(f"Saving Stage 2 model to {self.model_path}")
        joblib.dump(self.model, self.model_path)
        return self

    def predict(self, X):
        """
        Predicts using the trained LightGBM model.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading Stage 2 model from {self.model_path}")
                self.model = joblib.load(self.model_path)
            else:
                raise RuntimeError("Stage 2 model is not fitted and no cache found.")

        return self.model.predict(X)
