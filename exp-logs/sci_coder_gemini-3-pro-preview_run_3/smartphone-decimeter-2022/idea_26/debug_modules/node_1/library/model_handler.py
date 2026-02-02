import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.config import (
    WORKING_DIR,
    ML_FEATURES,
    TARGET_E,
    TARGET_N,
    LGBM_PARAMS,
    TRAIN_PARAMS,
    SEED,
)


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM training and inference.
    Manages separate models for East and North targets using GroupKFold CV.
    """

    def __init__(self):
        self.models_e = []
        self.models_n = []
        self.model_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train(
        self, train_df: pd.DataFrame, n_folds: int = 5, force_retrain: bool = True
    ):
        """
        Train LightGBM models for East and North targets.

        Parameters:
        -----------
        train_df : pd.DataFrame
            The training dataset containing features, targets, and drive_id.
        n_folds : int
            Number of folds for GroupKFold.
        force_retrain : bool
            If True, retrain models even if they exist on disk.
        """
        print(f"Starting training with {n_folds} folds...")

        # Reset models
        self.models_e = []
        self.models_n = []

        # Prepare data
        X = train_df[ML_FEATURES]
        y_e = train_df[TARGET_E]
        y_n = train_df[TARGET_N]
        groups = train_df["drive_id"]

        gkf = GroupKFold(n_splits=n_folds)

        oof_e = np.zeros(len(train_df))
        oof_n = np.zeros(len(train_df))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")

            X_train, y_e_train, y_n_train = (
                X.iloc[train_idx],
                y_e.iloc[train_idx],
                y_n.iloc[train_idx],
            )
            X_val, y_e_val, y_n_val = (
                X.iloc[val_idx],
                y_e.iloc[val_idx],
                y_n.iloc[val_idx],
            )

            # Train or Load East Model
            model_e_path = os.path.join(self.model_dir, f"lgbm_E_fold_{fold}.txt")
            if not force_retrain and os.path.exists(model_e_path):
                print(f"Loading East model from {model_e_path}")
                model_e = lgb.Booster(model_file=model_e_path)
            else:
                print("Training East Model...")
                dtrain_e = lgb.Dataset(X_train, label=y_e_train)
                dval_e = lgb.Dataset(X_val, label=y_e_val, reference=dtrain_e)

                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=TRAIN_PARAMS["early_stopping_rounds"]
                    ),
                    lgb.log_evaluation(period=TRAIN_PARAMS["verbose_eval"]),
                ]

                model_e = lgb.train(
                    LGBM_PARAMS,
                    dtrain_e,
                    valid_sets=[dtrain_e, dval_e],
                    valid_names=["train", "val"],
                    callbacks=callbacks,
                )
                model_e.save_model(model_e_path)

            self.models_e.append(model_e)
            oof_e[val_idx] = model_e.predict(
                X_val, num_iteration=model_e.best_iteration
            )

            # Train or Load North Model
            model_n_path = os.path.join(self.model_dir, f"lgbm_N_fold_{fold}.txt")
            if not force_retrain and os.path.exists(model_n_path):
                print(f"Loading North model from {model_n_path}")
                model_n = lgb.Booster(model_file=model_n_path)
            else:
                print("Training North Model...")
                dtrain_n = lgb.Dataset(X_train, label=y_n_train)
                dval_n = lgb.Dataset(X_val, label=y_n_val, reference=dtrain_n)

                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=TRAIN_PARAMS["early_stopping_rounds"]
                    ),
                    lgb.log_evaluation(period=TRAIN_PARAMS["verbose_eval"]),
                ]

                model_n = lgb.train(
                    LGBM_PARAMS,
                    dtrain_n,
                    valid_sets=[dtrain_n, dval_n],
                    valid_names=["train", "val"],
                    callbacks=callbacks,
                )
                model_n.save_model(model_n_path)

            self.models_n.append(model_n)
            oof_n[val_idx] = model_n.predict(
                X_val, num_iteration=model_n.best_iteration
            )

            # Fold Metrics
            mae_e = mean_absolute_error(y_e_val, oof_e[val_idx])
            mae_n = mean_absolute_error(y_n_val, oof_n[val_idx])
            print(
                f"Fold {fold + 1} MAE - East: {mae_e}, North: {mae_n}, Avg: {(mae_e + mae_n) / 2}"
            )

        # Overall Metrics
        total_mae_e = mean_absolute_error(y_e, oof_e)
        total_mae_n = mean_absolute_error(y_n, oof_n)
        print("\n=== Training Complete ===")
        print(f"Overall OOF MAE - East: {total_mae_e}")
        print(f"Overall OOF MAE - North: {total_mae_n}")
        print(f"Combined MAE: {(total_mae_e + total_mae_n) / 2}")

    def predict(self, test_df: pd.DataFrame, load_models_from_disk: bool = True):
        """
        Generate predictions for the test set using the trained ensemble.

        Parameters:
        -----------
        test_df : pd.DataFrame
            Test dataset containing features.
        load_models_from_disk : bool
            If True, loads models from self.model_dir. If False, uses models in memory.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns ['tripId', 'UnixTimeMillis', 'pred_E', 'pred_N']
        """
        print("Generating predictions...")

        # Load models if necessary
        if load_models_from_disk:
            self.models_e = []
            self.models_n = []

            # Find all model files
            files = os.listdir(self.model_dir)
            e_files = sorted([f for f in files if "lgbm_E_fold" in f])
            n_files = sorted([f for f in files if "lgbm_N_fold" in f])

            if not e_files or not n_files:
                raise FileNotFoundError("No trained models found in model directory.")

            print(f"Found {len(e_files)} East models and {len(n_files)} North models.")

            for f in e_files:
                self.models_e.append(
                    lgb.Booster(model_file=os.path.join(self.model_dir, f))
                )
            for f in n_files:
                self.models_n.append(
                    lgb.Booster(model_file=os.path.join(self.model_dir, f))
                )

        if not self.models_e or not self.models_n:
            raise ValueError("Models not loaded. Train or load from disk first.")

        X_test = test_df[ML_FEATURES]

        # East Predictions
        preds_e = np.zeros(len(X_test))
        for model in self.models_e:
            preds_e += model.predict(X_test, num_iteration=model.best_iteration)
        preds_e /= len(self.models_e)

        # North Predictions
        preds_n = np.zeros(len(X_test))
        for model in self.models_n:
            preds_n += model.predict(X_test, num_iteration=model.best_iteration)
        preds_n /= len(self.models_n)

        # Create result dataframe
        result = test_df[["tripId", "UnixTimeMillis"]].copy()
        result["pred_E"] = preds_e
        result["pred_N"] = preds_n

        return result
