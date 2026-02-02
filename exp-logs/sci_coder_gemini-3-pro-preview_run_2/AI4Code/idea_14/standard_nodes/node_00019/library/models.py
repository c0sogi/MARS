import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from library.config import Config


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor.
    Maps high-dimensional TF-IDF vectors directly to normalized ranks.
    Includes logic for generating Out-Of-Fold (OOF) predictions for stacking.
    """

    def __init__(self):
        self.model = Ridge(
            alpha=Config.RIDGE_ALPHA,
            random_state=Config.RIDGE_RANDOM_STATE,
            solver="auto",
        )
        self.model_path = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")

    def fit(self, X, y):
        """
        Fits the Ridge model on the provided data.

        Args:
            X (sparse matrix): TF-IDF features.
            y (array-like): Normalized ranks.
        """
        print("Training Stage 1 Ridge Regressor...")
        self.model.fit(X, y)
        print("Stage 1 training complete.")

    def predict(self, X):
        """
        Predicts normalized ranks.
        """
        return self.model.predict(X)

    def get_oof_predictions(self, X, y, groups=None, n_splits=5, load_cached_data=True):
        """
        Generates Out-Of-Fold predictions for the training set.
        Implements caching to avoid re-computation.

        Args:
            X (sparse matrix): Feature matrix.
            y (array-like): Target values.
            groups (array-like, optional): Group labels for splitting (not used in simple KFold but good for API).
            n_splits (int): Number of folds.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.array: OOF predictions aligned with X.
        """
        cache_path = os.path.join(Config.WORKING_DIR, "stage1_oof_preds.npy")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached OOF predictions from {cache_path}...")
                return np.load(cache_path)
            except Exception as e:
                print(f"Failed to load OOF cache ({e}). Recomputing...")

        # 2. Compute OOF
        print(f"Generating OOF predictions with {n_splits}-Fold CV...")
        kf = KFold(
            n_splits=n_splits, shuffle=True, random_state=Config.RIDGE_RANDOM_STATE
        )

        oof_preds = np.zeros(X.shape[0])

        # Ensure y is numpy array
        y = np.array(y)

        fold = 1
        for train_index, val_index in kf.split(X):
            X_train_fold = X[train_index]
            y_train_fold = y[train_index]
            X_val_fold = X[val_index]

            model_fold = Ridge(
                alpha=Config.RIDGE_ALPHA, random_state=Config.RIDGE_RANDOM_STATE
            )
            model_fold.fit(X_train_fold, y_train_fold)
            preds_fold = model_fold.predict(X_val_fold)

            oof_preds[val_index] = preds_fold

            # Simple metric print for the fold
            mae = np.mean(np.abs(preds_fold - y[val_index]))
            print(f"Fold {fold} MAE: {mae}")
            fold += 1

        # 3. Save Cache
        try:
            np.save(cache_path, oof_preds)
            print(f"Saved OOF predictions to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save OOF cache: {e}")

        return oof_preds

    def save(self):
        """Saves the trained model to disk."""
        joblib.dump(self.model, self.model_path)
        print(f"Stage 1 model saved to {self.model_path}")

    def load(self):
        """Loads the model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Stage 1 model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Stage 1 model not found at {self.model_path}")


class Stage2LGBM:
    """
    Stage 2: Multi-View Gradient Booster.
    Refines predictions using stacked features (Ridge OOF, Anchors, SVD).
    """

    def __init__(self):
        # Extract params from Config, ensure mutable copy
        self.params = Config.LGBM_PARAMS.copy()

        # Remove parameters that shouldn't be passed to constructor if they are fit params
        # But LGBMRegressor constructor accepts most. 'early_stopping_rounds' is often handled in fit callbacks in 4.x
        # We will handle early stopping in fit()
        if "early_stopping_rounds" in self.params:
            self.early_stopping_rounds = self.params.pop("early_stopping_rounds")
        else:
            self.early_stopping_rounds = 100

        self.model = LGBMRegressor(**self.params)
        self.model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the LightGBM model with Early Stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (array-like): Training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (array-like, optional): Validation targets.
        """
        print("Training Stage 2 LightGBM Regressor...")

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                early_stopping(
                    stopping_rounds=self.early_stopping_rounds, first_metric_only=True
                ),
                log_evaluation(period=100),
            ]

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )

        if X_val is not None and y_val is not None:
            # Print final metric
            if hasattr(self.model, "best_score_"):
                # Structure of best_score_: {'valid_0': {'l1': 0.1234}}
                print(f"Best Validation Score: {self.model.best_score_}")
            else:
                print("Training complete (no best_score_ attribute found).")

    def predict(self, X):
        """
        Predicts normalized ranks.
        """
        return self.model.predict(X)

    def save(self):
        """Saves the trained model to disk."""
        # LightGBM models can be saved as text or via joblib. Text is more portable.
        self.model.booster_.save_model(self.model_path)
        print(f"Stage 2 model saved to {self.model_path}")

    def load(self):
        """Loads the model from disk."""
        if os.path.exists(self.model_path):
            # We need to reconstruct the LGBMRegressor wrapper around the booster
            import lightgbm as lgb

            booster = lgb.Booster(model_file=self.model_path)
            self.model = LGBMRegressor(**self.params)
            # This is a bit of a hack to attach the booster to the sklearn wrapper
            self.model._Booster = booster
            self.model.fitted_ = True
            print(f"Stage 2 model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Stage 2 model not found at {self.model_path}")
