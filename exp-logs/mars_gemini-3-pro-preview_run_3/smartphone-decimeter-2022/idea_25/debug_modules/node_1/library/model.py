import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import (
    OUTPUT_DIR,
    ML_FEATURES,
    LGBM_PARAMS,
    N_FOLDS,
    EARLY_STOPPING_ROUNDS,
    SEED,
)


class LGBMEnsemble:
    """
    Ensemble of LightGBM regressors for predicting ENU position residuals.
    Trains separate models for Easting and Northing components using GroupKFold.
    """

    def __init__(self):
        self.models_E = []
        self.models_N = []
        self.model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, train_df, val_df=None):
        """
        Trains the ensemble using GroupKFold cross-validation.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame, optional): External validation data for evaluation.
        """
        # Prepare feature matrix and target vectors
        X = train_df[ML_FEATURES]
        y_E = train_df["res_E"]
        y_N = train_df["res_N"]
        groups = train_df["drive_id"]

        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=N_FOLDS)

        # Metrics storage
        mae_E_scores = []
        mae_N_scores = []

        print(f"Starting training with {N_FOLDS} folds...")

        # Prepare parameters (copy to avoid mutation)
        params = LGBM_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 2000)

        for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y_E, groups)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]

            # --- Train East Model ---
            y_E_train, y_E_valid = y_E.iloc[train_idx], y_E.iloc[valid_idx]

            dtrain_E = lgb.Dataset(X_train, label=y_E_train)
            dvalid_E = lgb.Dataset(X_valid, label=y_E_valid, reference=dtrain_E)

            model_E = lgb.train(
                params,
                dtrain_E,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid_E],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Save model
            model_E_path = os.path.join(self.model_dir, f"lgbm_E_fold_{fold}.txt")
            model_E.save_model(model_E_path)
            self.models_E.append(model_E)

            # Evaluate East
            pred_E_valid = model_E.predict(
                X_valid, num_iteration=model_E.best_iteration
            )
            mae_E = np.mean(np.abs(y_E_valid - pred_E_valid))
            mae_E_scores.append(mae_E)

            # --- Train North Model ---
            y_N_train, y_N_valid = y_N.iloc[train_idx], y_N.iloc[valid_idx]

            dtrain_N = lgb.Dataset(X_train, label=y_N_train)
            dvalid_N = lgb.Dataset(X_valid, label=y_N_valid, reference=dtrain_N)

            model_N = lgb.train(
                params,
                dtrain_N,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid_N],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Save model
            model_N_path = os.path.join(self.model_dir, f"lgbm_N_fold_{fold}.txt")
            model_N.save_model(model_N_path)
            self.models_N.append(model_N)

            # Evaluate North
            pred_N_valid = model_N.predict(
                X_valid, num_iteration=model_N.best_iteration
            )
            mae_N = np.mean(np.abs(y_N_valid - pred_N_valid))
            mae_N_scores.append(mae_N)

            print(f"Fold {fold}: MAE_E = {mae_E}, MAE_N = {mae_N}")

        # Print Overall Metrics
        print(f"Average MAE_E: {np.mean(mae_E_scores)}")
        print(f"Average MAE_N: {np.mean(mae_N_scores)}")

        # Validation on external val set if provided
        if val_df is not None and not val_df.empty:
            print("Evaluating on external validation set...")
            preds = self.predict(val_df)
            val_mae_E = np.mean(np.abs(val_df["res_E"] - preds["pred_E"]))
            val_mae_N = np.mean(np.abs(val_df["res_N"] - preds["pred_N"]))
            print(f"External Val MAE_E: {val_mae_E}")
            print(f"External Val MAE_N: {val_mae_N}")

    def predict(self, test_df):
        """
        Generates predictions for the test set using the trained ensemble.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame with 'UnixTimeMillis', 'pred_E', 'pred_N'.
        """
        X_test = test_df[ML_FEATURES]

        pred_E = np.zeros(len(test_df))
        pred_N = np.zeros(len(test_df))

        # Ensure models are loaded
        if not self.models_E or not self.models_N:
            # Fallback: try loading from disk if memory is empty
            # This handles the case where we might want to predict without retraining in the same session
            # though the current pipeline flow usually trains then predicts.
            for fold in range(N_FOLDS):
                e_path = os.path.join(self.model_dir, f"lgbm_E_fold_{fold}.txt")
                n_path = os.path.join(self.model_dir, f"lgbm_N_fold_{fold}.txt")
                if os.path.exists(e_path):
                    self.models_E.append(lgb.Booster(model_file=e_path))
                if os.path.exists(n_path):
                    self.models_N.append(lgb.Booster(model_file=n_path))

        if not self.models_E or not self.models_N:
            raise RuntimeError("No trained models found for prediction.")

        # Average predictions from all fold models (East)
        for model in self.models_E:
            pred_E += model.predict(X_test, num_iteration=model.best_iteration)
        pred_E /= len(self.models_E)

        # Average predictions from all fold models (North)
        for model in self.models_N:
            pred_N += model.predict(X_test, num_iteration=model.best_iteration)
        pred_N /= len(self.models_N)

        result_df = pd.DataFrame(
            {
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "pred_E": pred_E,
                "pred_N": pred_N,
            }
        )

        return result_df
