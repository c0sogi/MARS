import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import (
    SEED,
    N_FOLDS,
    N_JOBS,
    LGBM_PARAMS,
    XGB_PARAMS,
    HGB_PARAMS,
    RIDGE_PARAMS,
)
from library.utils import seed_everything, calculate_mae
from library.model_definitions import (
    get_lgbm_regressor,
    get_xgb_regressor,
    get_catboost_regressor,
    get_meta_learner,
)


class StackingTrainer:
    """
    Manages the training, validation, and inference of a Two-Level Stacking Ensemble.
    Level 0: LightGBM, XGBoost, HistGradientBoosting
    Level 1: Ridge Regression
    """

    def __init__(self):
        seed_everything(SEED)
        self.base_models = {}
        self.meta_model = None
        self.best_iterations = {"lgbm": [], "xgb": []}

    def train_base_layer(self, X, y):
        """
        Trains base models using Stratified K-Fold CV and generates OOF predictions.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            np.ndarray: OOF predictions matrix (n_samples, 3).
        """
        # Bin targets for stratified split
        num_bins = int(np.floor(1 + np.log2(len(y))))
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof_preds = np.zeros((X.shape[0], 3))

        print(f"Starting {N_FOLDS}-Fold CV Base Layer Training...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # --- Model 1: LightGBM ---
            # Note: Using callbacks for early stopping as per recent LightGBM API
            model_lgb = get_lgbm_regressor()

            callbacks = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            model_lgb.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=callbacks,
            )

            # Predict
            pred_lgb = model_lgb.predict(X_val)
            oof_preds[val_idx, 0] = pred_lgb

            # Store best iteration
            if hasattr(model_lgb, "best_iteration_"):
                self.best_iterations["lgbm"].append(model_lgb.best_iteration_)

            # --- Model 2: XGBoost ---
            model_xgb = get_xgb_regressor(early_stopping_rounds=100)
            # XGBoost sklearn API supports early_stopping_rounds in fit
            model_xgb.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            pred_xgb = model_xgb.predict(X_val)
            oof_preds[val_idx, 1] = pred_xgb

            # Store best iteration
            if hasattr(model_xgb, "best_iteration"):
                self.best_iterations["xgb"].append(model_xgb.best_iteration)

            # --- Model 3: HistGradientBoosting ---
            # HGB handles validation internally via validation_fraction in params
            model_hgb = get_catboost_regressor()
            model_hgb.fit(X_train, y_train)

            pred_hgb = model_hgb.predict(X_val)
            oof_preds[val_idx, 2] = pred_hgb

            # Calculate Scores
            mae_lgb = calculate_mae(y_val, pred_lgb)
            mae_xgb = calculate_mae(y_val, pred_xgb)
            mae_hgb = calculate_mae(y_val, pred_hgb)

            print(f"Fold {fold + 1} MAE - LGBM: {mae_lgb}")
            print(f"Fold {fold + 1} MAE - XGB: {mae_xgb}")
            print(f"Fold {fold + 1} MAE - HGB: {mae_hgb}")

        return oof_preds

    def train_meta_layer(self, oof_preds, y):
        """
        Trains the Meta Learner (Ridge) on OOF predictions.

        Args:
            oof_preds (np.ndarray): Matrix of OOF predictions from base layer.
            y (np.ndarray): True targets.
        """
        print("Training Meta Learner (Ridge)...")
        self.meta_model = get_meta_learner()
        self.meta_model.fit(oof_preds, y)

        # Evaluate on OOF (Proxy for CV score)
        meta_preds = self.meta_model.predict(oof_preds)
        score = calculate_mae(y, meta_preds)
        print(f"Final Stacking CV MAE: {score}")

    def retrain_full_base(self, X, y):
        """
        Retrains base models on the full dataset using average best iterations.

        Args:
            X (np.ndarray): Full feature matrix.
            y (np.ndarray): Full target vector.
        """
        print("Retraining Base Learners on Full Data...")

        # --- LightGBM ---
        avg_iter_lgbm = int(np.mean(self.best_iterations["lgbm"]))
        print(f"Retraining LGBM with {avg_iter_lgbm} estimators.")
        self.base_models["lgbm"] = get_lgbm_regressor(n_estimators=avg_iter_lgbm)
        self.base_models["lgbm"].fit(X, y)

        # --- XGBoost ---
        avg_iter_xgb = int(np.mean(self.best_iterations["xgb"]))
        print(f"Retraining XGBoost with {avg_iter_xgb} estimators.")
        # Disable early stopping for full training by not providing eval_set
        # and setting n_estimators explicitly
        self.base_models["xgb"] = get_xgb_regressor(
            n_estimators=avg_iter_xgb, early_stopping_rounds=None
        )
        self.base_models["xgb"].fit(X, y, verbose=False)

        # --- HistGradientBoosting ---
        # HGB doesn't support exact iteration setting easily via public API
        # without disabling its internal logic, so we retrain with default/config params
        print("Retraining HistGradientBoosting...")
        self.base_models["hgb"] = get_catboost_regressor()
        self.base_models["hgb"].fit(X, y)

    def predict_stack(self, X_test):
        """
        Generates predictions for the test set using the stacked ensemble.

        Args:
            X_test (np.ndarray): Test feature matrix.

        Returns:
            np.ndarray: Final predictions.
        """
        print("Generating Test Predictions...")

        # Base Layer Predictions
        pred_lgb = self.base_models["lgbm"].predict(X_test)
        pred_xgb = self.base_models["xgb"].predict(X_test)
        pred_hgb = self.base_models["hgb"].predict(X_test)

        # Stack
        base_preds = np.column_stack([pred_lgb, pred_xgb, pred_hgb])

        # Meta Layer Prediction
        final_preds = self.meta_model.predict(base_preds)

        return final_preds
