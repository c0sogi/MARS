import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from typing import List, Tuple, Optional, Dict, Any

from library.config import LGBM_PARAMS, XGB_PARAMS, WORKING_DIR, SEED
from library.utils import seed_everything


class UnifiedEnsemble:
    """
    A heterogeneous ensemble model combining LightGBM and XGBoost.
    Designed for the NFL Contact Detection task.
    """

    def __init__(self):
        """
        Initialize the ensemble with placeholders for the models.
        """
        self.lgbm_model = None
        self.xgb_model = None
        seed_everything(SEED)

    def train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_cols: List[str],
    ) -> lgb.Booster:
        """
        Trains the LightGBM model with early stopping.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            feature_cols: List of feature column names to use.

        Returns:
            Trained LightGBM Booster.
        """
        print("Training LightGBM...")

        # Prepare Datasets
        dtrain = lgb.Dataset(X_train[feature_cols], label=y_train)
        dval = lgb.Dataset(X_val[feature_cols], label=y_val, reference=dtrain)

        # Extract n_estimators for num_boost_round and remove from params
        params = LGBM_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 2000)

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        # Train
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log best score
        best_score = model.best_score["valid"]["auc"]
        print(f"LightGBM Best Validation AUC: {best_score:.10f}")

        return model

    def train_xgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_cols: List[str],
    ) -> xgb.Booster:
        """
        Trains the XGBoost model with early stopping.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            feature_cols: List of feature column names to use.

        Returns:
            Trained XGBoost Booster.
        """
        print("Training XGBoost...")

        # Prepare DMatrices
        dtrain = xgb.DMatrix(X_train[feature_cols], label=y_train)
        dval = xgb.DMatrix(X_val[feature_cols], label=y_val)

        # Extract n_estimators and remove from params
        params = XGB_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 2000)

        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Log best score (XGBoost stores best_score as a float attribute if early stopping is used)
        # However, accessing the specific metric value programmatically can be tricky across versions.
        # We rely on the verbose output for the log, but we can try to fetch it.
        print(f"XGBoost Best Iteration: {model.best_iteration}")
        # Note: XGBoost python API doesn't always expose best_score directly in the object
        # the same way LGBM does, but it uses best_ntree_limit for prediction.

        return model

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "contact",
    ):
        """
        Fits both models in the ensemble.

        Args:
            df_train: Training dataframe containing features and target.
            df_val: Validation dataframe containing features and target.
            feature_cols: List of feature column names.
            target_col: Name of the target column.
        """
        X_train = df_train
        y_train = df_train[target_col]
        X_val = df_val
        y_val = df_val[target_col]

        # Train LightGBM
        self.lgbm_model = self.train_lgbm(X_train, y_train, X_val, y_val, feature_cols)

        # Train XGBoost
        self.xgb_model = self.train_xgb(X_train, y_train, X_val, y_val, feature_cols)

        # Save models immediately after training
        self.save_models()

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Generates predictions using the ensemble (unweighted average).

        Args:
            df: Dataframe containing features.
            feature_cols: List of feature column names.

        Returns:
            Numpy array of probabilities.
        """
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("Models have not been trained or loaded.")

        # LightGBM Prediction
        pred_lgbm = self.lgbm_model.predict(
            df[feature_cols], num_iteration=self.lgbm_model.best_iteration
        )

        # XGBoost Prediction
        dtest = xgb.DMatrix(df[feature_cols])
        pred_xgb = self.xgb_model.predict(
            dtest, iteration_range=(0, self.xgb_model.best_iteration + 1)
        )

        # Ensemble (Unweighted Average)
        pred_ensemble = (pred_lgbm + pred_xgb) / 2.0

        return pred_ensemble

    def save_models(self, suffix: str = ""):
        """
        Saves the trained models to the working directory.

        Args:
            suffix: Optional suffix for filenames (e.g., '_fold1').
        """
        if self.lgbm_model:
            path = os.path.join(WORKING_DIR, f"lgbm_model{suffix}.joblib")
            joblib.dump(self.lgbm_model, path)
            print(f"Saved LightGBM model to {path}")

        if self.xgb_model:
            path = os.path.join(WORKING_DIR, f"xgb_model{suffix}.joblib")
            joblib.dump(self.xgb_model, path)
            print(f"Saved XGBoost model to {path}")

    def load_models(self, suffix: str = ""):
        """
        Loads trained models from the working directory.

        Args:
            suffix: Optional suffix for filenames.
        """
        lgbm_path = os.path.join(WORKING_DIR, f"lgbm_model{suffix}.joblib")
        xgb_path = os.path.join(WORKING_DIR, f"xgb_model{suffix}.joblib")

        if os.path.exists(lgbm_path):
            self.lgbm_model = joblib.load(lgbm_path)
            print(f"Loaded LightGBM model from {lgbm_path}")
        else:
            print(f"Warning: LightGBM model not found at {lgbm_path}")

        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            print(f"Loaded XGBoost model from {xgb_path}")
        else:
            print(f"Warning: XGBoost model not found at {xgb_path}")
