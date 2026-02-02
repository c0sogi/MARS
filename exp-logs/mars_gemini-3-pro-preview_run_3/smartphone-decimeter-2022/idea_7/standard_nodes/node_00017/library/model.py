import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import (
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    WORKING_DIR,
)


class ResidualBooster:
    """
    A wrapper class for training and predicting with LightGBM models
    specifically for ENU residual regression.
    """

    def __init__(self):
        self.models_east = []
        self.models_north = []
        self.feature_names = []

    def train_cv(
        self, X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series, n_splits: int = 5
    ):
        """
        Perform GroupKFold cross-validation training.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.DataFrame): Target dataframe with 'target_e' and 'target_n'.
            groups (pd.Series): Group identifiers for GroupKFold (e.g., drive_id).
            n_splits (int): Number of cross-validation splits.
        """
        self.models_east = []
        self.models_north = []
        self.feature_names = X.columns.tolist()

        gkf = GroupKFold(n_splits=n_splits)

        fold_scores_e = []
        fold_scores_n = []

        print(f"Starting GroupKFold Cross-Validation with {n_splits} splits...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\n--- Fold {fold + 1} ---")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # -------------------------
            # Train East Model
            # -------------------------
            print("Training East Model...")
            dtrain_e = lgb.Dataset(X_train, label=y_train["target_e"])
            dval_e = lgb.Dataset(X_val, label=y_val["target_e"], reference=dtrain_e)

            callbacks_e = [
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=VERBOSE_EVAL),
            ]

            model_e = lgb.train(
                LGBM_PARAMS,
                dtrain_e,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dtrain_e, dval_e],
                valid_names=["train", "valid"],
                callbacks=callbacks_e,
            )

            # Save model to text file
            model_e_path = os.path.join(WORKING_DIR, f"lgbm_east_fold_{fold}.txt")
            model_e.save_model(model_e_path)
            self.models_east.append(model_e)

            # Evaluate East
            preds_e = model_e.predict(X_val, num_iteration=model_e.best_iteration)
            mae_e = np.mean(np.abs(y_val["target_e"] - preds_e))
            fold_scores_e.append(mae_e)
            print(f"Fold {fold + 1} East MAE: {mae_e}")

            # -------------------------
            # Train North Model
            # -------------------------
            print("Training North Model...")
            dtrain_n = lgb.Dataset(X_train, label=y_train["target_n"])
            dval_n = lgb.Dataset(X_val, label=y_val["target_n"], reference=dtrain_n)

            callbacks_n = [
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=VERBOSE_EVAL),
            ]

            model_n = lgb.train(
                LGBM_PARAMS,
                dtrain_n,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dtrain_n, dval_n],
                valid_names=["train", "valid"],
                callbacks=callbacks_n,
            )

            # Save model to text file
            model_n_path = os.path.join(WORKING_DIR, f"lgbm_north_fold_{fold}.txt")
            model_n.save_model(model_n_path)
            self.models_north.append(model_n)

            # Evaluate North
            preds_n = model_n.predict(X_val, num_iteration=model_n.best_iteration)
            mae_n = np.mean(np.abs(y_val["target_n"] - preds_n))
            fold_scores_n.append(mae_n)
            print(f"Fold {fold + 1} North MAE: {mae_n}")

        # Summary
        mean_mae_e = np.mean(fold_scores_e)
        mean_mae_n = np.mean(fold_scores_n)
        mean_mae_total = (mean_mae_e + mean_mae_n) / 2

        print("\n" + "=" * 40)
        print("Cross-Validation Results")
        print("=" * 40)
        print(f"Mean East MAE:  {mean_mae_e}")
        print(f"Mean North MAE: {mean_mae_n}")
        print(f"Average MAE:    {mean_mae_total}")
        print("=" * 40)

    def predict(self, X: pd.DataFrame):
        """
        Generate predictions by averaging outputs from all fold models.

        Args:
            X (pd.DataFrame): Features for prediction.

        Returns:
            tuple: (pred_e, pred_n) - Numpy arrays of predicted residuals.
        """
        if not self.models_east or not self.models_north:
            # Try to load from disk if not in memory
            print("Models not found in memory. Attempting to load from disk...")
            self.models_east = []
            self.models_north = []

            # Assume 5 folds as default if not specified, or scan directory
            # Scanning directory for model files
            import glob

            east_files = sorted(
                glob.glob(os.path.join(WORKING_DIR, "lgbm_east_fold_*.txt"))
            )
            north_files = sorted(
                glob.glob(os.path.join(WORKING_DIR, "lgbm_north_fold_*.txt"))
            )

            for f in east_files:
                self.models_east.append(lgb.Booster(model_file=f))
            for f in north_files:
                self.models_north.append(lgb.Booster(model_file=f))

            if not self.models_east or not self.models_north:
                raise RuntimeError("No trained models found. Call train_cv first.")

        # Predict East
        preds_e = np.zeros(len(X))
        for model in self.models_east:
            preds_e += model.predict(X, num_iteration=model.best_iteration)
        preds_e /= len(self.models_east)

        # Predict North
        preds_n = np.zeros(len(X))
        for model in self.models_north:
            preds_n += model.predict(X, num_iteration=model.best_iteration)
        preds_n /= len(self.models_north)

        return preds_e, preds_n
