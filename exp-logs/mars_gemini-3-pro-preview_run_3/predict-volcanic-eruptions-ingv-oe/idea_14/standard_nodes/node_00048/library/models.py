import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything, compute_mae

# Ensure reproducibility
seed_everything(Config.SEED)


class BaseModelWrapper:
    """
    A unified wrapper for LightGBM, XGBoost, and CatBoost to handle
    differences in API, early stopping, and best iteration retrieval.
    """

    def __init__(self, model_type, params):
        self.model_type = model_type
        self.params = params.copy()
        self.model = None
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None, fixed_iterations=None):
        """
        Fits the model.
        If X_val/y_val are provided, uses early stopping.
        If fixed_iterations is provided, runs for that specific number of rounds (used for full retrain).
        """
        # Handle fixed iterations for full retraining
        if fixed_iterations is not None:
            if self.model_type == "lgbm":
                self.params["n_estimators"] = int(fixed_iterations)
            elif self.model_type == "xgb":
                self.params["n_estimators"] = int(fixed_iterations)
            elif self.model_type == "cat":
                self.params["iterations"] = int(fixed_iterations)

        # Initialize model
        if self.model_type == "lgbm":
            self.model = lgb.LGBMRegressor(**self.params)
        elif self.model_type == "xgb":
            # For XGBoost 3.x, early_stopping_rounds must be in constructor
            if X_val is not None and y_val is not None and fixed_iterations is None:
                self.params["early_stopping_rounds"] = Config.EARLY_STOPPING_ROUNDS
            self.model = xgb.XGBRegressor(**self.params)
        elif self.model_type == "cat":
            self.model = CatBoostRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fit with Early Stopping if validation data is present
        if X_val is not None and y_val is not None and fixed_iterations is None:
            if self.model_type == "lgbm":
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Suppress logging
                ]
                self.model.fit(
                    X,
                    y,
                    eval_set=[(X_val, y_val)],
                    eval_metric="mae",
                    callbacks=callbacks,
                )
                self.best_iteration = self.model.best_iteration_

            elif self.model_type == "xgb":
                self.model.fit(
                    X,
                    y,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                self.best_iteration = self.model.best_iteration

            elif self.model_type == "cat":
                self.model.fit(
                    X,
                    y,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
                self.best_iteration = self.model.get_best_iteration()

        else:
            # Full training without validation (using fixed iterations or default)
            self.model.fit(X, y)
            self.best_iteration = None  # Not applicable

    def predict(self, X):
        return self.model.predict(X)


class LGBMTrainer:
    """
    Manages a single LightGBM model pipeline.
    Handles CV for optimal iterations and full retraining.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS
        self.model_full = None
        self.optimal_iterations = []
        self.model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.pkl")

    def _get_stratified_folds(self, y, n_splits):
        """
        Creates stratified folds for continuous target by binning.
        """
        num_bins = min(10, len(np.unique(y)))
        if num_bins < 2:
            from sklearn.model_selection import KFold

            return KFold(
                n_splits=n_splits, shuffle=True, random_state=Config.SEED
            ).split(np.zeros(len(y)), y)

        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        return skf.split(np.zeros(len(y)), y_bins)

    def fit_pipeline(self, X, y):
        """
        Executes the training pipeline:
        1. CV to find optimal iterations.
        2. Retraining on full dataset.
        """
        print(f"Starting LightGBM CV...")
        oof_preds = np.zeros(len(X))
        folds = list(self._get_stratified_folds(y, Config.N_FOLDS))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"  Processing Fold {fold_idx + 1}/{Config.N_FOLDS}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = BaseModelWrapper("lgbm", self.params)
            model.fit(X_train, y_train, X_val, y_val)

            if model.best_iteration is not None:
                self.optimal_iterations.append(model.best_iteration)

            oof_preds[val_idx] = model.predict(X_val)

        mae = compute_mae(y, oof_preds)
        print(f"\nLightGBM OOF MAE: {mae}")

        # Retrain on full
        print("\nRetraining LightGBM on full dataset...")
        if self.optimal_iterations:
            avg_iter = max(1, int(np.mean(self.optimal_iterations)))
            print(f"  Retraining with {avg_iter} iterations (Avg from CV)")
        else:
            avg_iter = Config.N_ESTIMATORS
            print(f"  Retraining with default {avg_iter} iterations")

        self.model_full = BaseModelWrapper("lgbm", self.params)
        self.model_full.fit(X, y, fixed_iterations=avg_iter)

        self.save_models()

    def predict(self, X):
        if self.model_full is None:
            raise RuntimeError("Model not trained.")
        return self.model_full.predict(X)

    def save_models(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_models(self):
        if os.path.exists(self.model_path):
            loaded_obj = joblib.load(self.model_path)
            self.__dict__.update(loaded_obj.__dict__)
            return True
        return False

    def save_models(self):
        """Saves the StackingManager state to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self, self.model_path)
        print(f"Models saved to {self.model_path}")

    def load_models(self):
        """Loads the StackingManager state from disk."""
        if os.path.exists(self.model_path):
            loaded_obj = joblib.load(self.model_path)
            self.__dict__.update(loaded_obj.__dict__)
            print(f"Models loaded from {self.model_path}")
            return True
        else:
            print(f"No saved models found at {self.model_path}")
            return False
