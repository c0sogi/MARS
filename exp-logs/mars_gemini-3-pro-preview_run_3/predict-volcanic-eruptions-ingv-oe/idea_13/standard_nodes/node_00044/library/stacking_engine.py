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
    Manages the training, validation, and inference.
    Optimized to use a Single Strong Learner (LightGBM) instead of a complex stack.
    Cite Lesson 00031: Single strong GBDT often outperforms complex ensembles in this domain.
    """

    def __init__(self):
        seed_everything(SEED)
        self.model = None
        self.best_iterations = []

    def train_base_layer(self, X, y):
        """
        Trains LightGBM using Stratified K-Fold CV.
        Returns OOF predictions.
        """
        # Bin targets for stratified split
        num_bins = int(np.floor(1 + np.log2(len(y))))
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof_preds = np.zeros(X.shape[0])

        print(f"Starting {N_FOLDS}-Fold CV Training (Single LightGBM)...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # --- LightGBM ---
            model_lgb = get_lgbm_regressor()

            callbacks = [
                lgb.early_stopping(stopping_rounds=200, verbose=False),
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
            oof_preds[val_idx] = pred_lgb

            # Store best iteration
            if hasattr(model_lgb, "best_iteration_"):
                self.best_iterations.append(model_lgb.best_iteration_)

            mae_lgb = calculate_mae(y_val, pred_lgb)
            print(f"Fold {fold + 1} MAE - LGBM: {mae_lgb}")

        return oof_preds

    def train_meta_layer(self, oof_preds, y):
        """
        No-op for single model pipeline.
        Just prints the final CV score.
        """
        score = calculate_mae(y, oof_preds)
        print(f"Final CV MAE: {score}")

    def retrain_full_base(self, X, y):
        """
        Retrains the single LightGBM on the full dataset.
        """
        print("Retraining LightGBM on Full Data...")
        avg_iter = int(np.mean(self.best_iterations))
        print(f"Retraining with {avg_iter} estimators.")

        self.model = get_lgbm_regressor(n_estimators=avg_iter)
        self.model.fit(X, y)

    def predict_stack(self, X_test):
        """
        Generates predictions using the single LightGBM model.
        """
        print("Generating Predictions...")
        return self.model.predict(X_test)
