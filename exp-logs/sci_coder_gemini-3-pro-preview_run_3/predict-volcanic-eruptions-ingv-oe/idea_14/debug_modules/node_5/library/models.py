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


class StackingManager:
    """
    Manages the Multi-View Dual-Stream Stacking pipeline.
    Handles Level 0 (Base Learners) CV, Level 1 (Meta Learner) training,
    and final full-dataset retraining.
    """

    def __init__(self):
        self.base_model_types = ["lgbm", "xgb", "cat"]
        self.base_params = {
            "lgbm": Config.LGBM_PARAMS,
            "xgb": Config.XGB_PARAMS,
            "cat": Config.CATBOOST_PARAMS,
        }

        # Storage for trained models
        self.level0_models_full = {}  # Dictionary to store final retrained models
        self.level1_model = None

        # Storage for optimal iterations found during CV
        self.optimal_iterations = {k: [] for k in self.base_model_types}

        # Paths
        self.model_path = os.path.join(Config.WORKING_DIR, "stacking_manager.pkl")

    def _get_stratified_folds(self, y, n_splits):
        """
        Creates stratified folds for continuous target by binning.
        """
        # Create bins for stratification
        num_bins = min(10, len(np.unique(y)))
        if num_bins < 2:
            # Fallback to KFold if not enough unique values
            from sklearn.model_selection import KFold

            return KFold(
                n_splits=n_splits, shuffle=True, random_state=Config.SEED
            ).split(np.zeros(len(y)), y)

        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        return skf.split(np.zeros(len(y)), y_bins)

    def train_level0_cv(self, X, y):
        """
        Level 0: Train base learners using Stratified K-Fold CV.
        Generates Out-of-Fold (OOF) predictions and records best iterations.
        """
        print(f"Starting Level 0 CV with {len(self.base_model_types)} base models...")

        oof_preds = pd.DataFrame(index=X.index)

        # Initialize OOF columns
        for m_type in self.base_model_types:
            oof_preds[f"pred_{m_type}"] = 0.0

        # CV Loop
        folds = list(self._get_stratified_folds(y, Config.N_FOLDS))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"  Processing Fold {fold_idx + 1}/{Config.N_FOLDS}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            for m_type in self.base_model_types:
                # Initialize wrapper
                model = BaseModelWrapper(m_type, self.base_params[m_type])

                # Fit with early stopping
                model.fit(X_train, y_train, X_val, y_val)

                # Record best iteration
                if model.best_iteration is not None:
                    self.optimal_iterations[m_type].append(model.best_iteration)

                # Predict OOF
                pred = model.predict(X_val)
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc(f"pred_{m_type}")] = (
                    pred
                )

        # Calculate and print OOF scores
        print("\nLevel 0 OOF Scores (MAE):")
        for m_type in self.base_model_types:
            mae = compute_mae(y, oof_preds[f"pred_{m_type}"])
            print(f"  {m_type.upper()}: {mae}")

        return oof_preds

    def train_level1(self, oof_preds, y):
        """
        Level 1: Train Meta Learner (Ridge) on OOF predictions.
        """
        print("\nTraining Level 1 Meta Learner (Ridge)...")
        self.level1_model = Ridge(**Config.RIDGE_PARAMS)
        self.level1_model.fit(oof_preds, y)

        # Check in-sample score of stacker (on OOF data)
        stack_preds = self.level1_model.predict(oof_preds)
        stack_mae = compute_mae(y, stack_preds)
        print(f"  Stacked OOF MAE: {stack_mae}")

    def retrain_level0_full(self, X, y):
        """
        Retrain Level 0 models on the full dataset using average best iterations from CV.
        """
        print("\nRetraining Level 0 models on full dataset...")

        for m_type in self.base_model_types:
            # Determine optimal iterations
            if self.optimal_iterations[m_type]:
                avg_iter = max(1, int(np.mean(self.optimal_iterations[m_type])))
                print(
                    f"  {m_type.upper()}: Retraining with {avg_iter} iterations (Avg from CV)"
                )
            else:
                avg_iter = Config.N_ESTIMATORS
                print(
                    f"  {m_type.upper()}: Retraining with default {avg_iter} iterations"
                )

            # Initialize and fit
            model = BaseModelWrapper(m_type, self.base_params[m_type])
            model.fit(X, y, fixed_iterations=avg_iter)

            self.level0_models_full[m_type] = model

    def fit_pipeline(self, X, y):
        """
        Executes the full training pipeline:
        1. Level 0 CV -> OOF Preds & Optimal Iterations
        2. Level 1 Training -> Meta Learner
        3. Level 0 Retraining -> Final Base Models
        """
        # 1. Level 0 CV
        oof_preds = self.train_level0_cv(X, y)

        # 2. Level 1 Training
        self.train_level1(oof_preds, y)

        # 3. Level 0 Full Retrain
        self.retrain_level0_full(X, y)

        # Save complete pipeline
        self.save_models()

    def predict(self, X):
        """
        Generates predictions for new data.
        1. Generate base predictions from Level 0 models.
        2. Feed base predictions to Level 1 model.
        """
        if not self.level0_models_full or self.level1_model is None:
            raise RuntimeError("Models not trained or loaded.")

        # Generate base predictions
        base_preds = pd.DataFrame(index=X.index)
        for m_type in self.base_model_types:
            model = self.level0_models_full[m_type]
            base_preds[f"pred_{m_type}"] = model.predict(X)

        # Meta prediction
        final_preds = self.level1_model.predict(base_preds)
        return final_preds

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
