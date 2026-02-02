import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import (
    WORKING_DIR,
    LGBM_PARAMS,
    N_FOLDS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SEED,
)


class LGBMEnsemble:
    """
    Ensemble of LightGBM regressors for East and North error prediction.
    Uses Unified Geometric Projection features.
    """

    def __init__(self):
        self.models_e = []
        self.models_n = []
        self.model_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        # Features defined in feature_eng.py
        self.feature_cols = [
            "F_pr_E",
            "F_pr_N",
            "F_pr_U",
            "F_dop_E",
            "F_dop_N",
            "F_dop_U",
            "G_xx",
            "G_yy",
            "G_zz",
            "Cn0_mean",
            "Sv_count",
        ]

    def _get_model_paths(self, fold):
        """Returns paths for East and North models for a specific fold."""
        return (
            os.path.join(self.model_dir, f"lgbm_E_fold_{fold}.txt"),
            os.path.join(self.model_dir, f"lgbm_N_fold_{fold}.txt"),
        )

    def fit(self, train_df, load_cached_data=True):
        """
        Trains the ensemble using GroupKFold cross-validation.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            load_cached_data (bool): If True, attempts to load trained models from disk.
        """
        # Check if all models exist for caching
        all_exist = True
        for fold in range(N_FOLDS):
            p_e, p_n = self._get_model_paths(fold)
            if not (os.path.exists(p_e) and os.path.exists(p_n)):
                all_exist = False
                break

        if load_cached_data and all_exist:
            print("Loading cached models from disk...")
            self.models_e = []
            self.models_n = []
            for fold in range(N_FOLDS):
                p_e, p_n = self._get_model_paths(fold)
                self.models_e.append(lgb.Booster(model_file=p_e))
                self.models_n.append(lgb.Booster(model_file=p_n))
            return

        print(f"Training LightGBM Ensemble with {N_FOLDS} folds...")

        # Prepare Feature Matrix and Targets
        X = train_df[self.feature_cols]
        y_e = train_df["Target_E"]
        y_n = train_df["Target_N"]

        # Extract groups (drive_id) from tripId
        # tripId format: drive_id-phone_name
        # We assume the last part is phone_name, rest is drive_id
        groups = train_df["tripId"].apply(lambda x: "-".join(x.split("-")[:-1]))

        gkf = GroupKFold(n_splits=N_FOLDS)

        self.models_e = []
        self.models_n = []

        # Prepare params (copy to avoid modification)
        params = LGBM_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 5000)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
            print(f"\n--- Fold {fold + 1} ---")

            X_train = X.iloc[train_idx]
            y_e_train = y_e.iloc[train_idx]
            y_n_train = y_n.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_e_val = y_e.iloc[val_idx]
            y_n_val = y_n.iloc[val_idx]

            # --- Train East Model ---
            print("Training East Model...")
            dtrain_e = lgb.Dataset(X_train, label=y_e_train)
            dval_e = lgb.Dataset(X_val, label=y_e_val, reference=dtrain_e)

            model_e = lgb.train(
                params,
                dtrain_e,
                valid_sets=[dtrain_e, dval_e],
                valid_names=["train", "val"],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(VERBOSE_EVAL),
                ],
            )
            self.models_e.append(model_e)

            # Manual Metric Print (Full Precision)
            pred_e_val = model_e.predict(X_val, num_iteration=model_e.best_iteration)
            mae_e = np.mean(np.abs(y_e_val - pred_e_val))
            print(f"Fold {fold + 1} East MAE: {mae_e}")

            # --- Train North Model ---
            print("Training North Model...")
            dtrain_n = lgb.Dataset(X_train, label=y_n_train)
            dval_n = lgb.Dataset(X_val, label=y_n_val, reference=dtrain_n)

            model_n = lgb.train(
                params,
                dtrain_n,
                valid_sets=[dtrain_n, dval_n],
                valid_names=["train", "val"],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(VERBOSE_EVAL),
                ],
            )
            self.models_n.append(model_n)

            # Manual Metric Print (Full Precision)
            pred_n_val = model_n.predict(X_val, num_iteration=model_n.best_iteration)
            mae_n = np.mean(np.abs(y_n_val - pred_n_val))
            print(f"Fold {fold + 1} North MAE: {mae_n}")

            # Save models
            p_e, p_n = self._get_model_paths(fold)
            model_e.save_model(p_e)
            model_n.save_model(p_n)

    def predict(self, test_df):
        """
        Generates predictions using the trained ensemble.

        Args:
            test_df (pd.DataFrame): Dataframe containing features.

        Returns:
            tuple: (pred_e, pred_n) - Averaged predictions for East and North errors.
        """
        if not self.models_e or not self.models_n:
            raise RuntimeError("Models not trained or loaded. Call fit() first.")

        X_test = test_df[self.feature_cols]

        pred_e = np.zeros(len(test_df))
        pred_n = np.zeros(len(test_df))

        # Average predictions across folds
        for model in self.models_e:
            pred_e += model.predict(X_test, num_iteration=model.best_iteration)
        pred_e /= len(self.models_e)

        for model in self.models_n:
            pred_n += model.predict(X_test, num_iteration=model.best_iteration)
        pred_n /= len(self.models_n)

        return pred_e, pred_n
