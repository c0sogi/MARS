import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import (
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
)


class LGBMModel:
    """
    Wrapper for LightGBM models to predict ENU residuals.
    Trains separate models for East and North components using the GPR-Boost strategy.
    """

    def __init__(self, params=None):
        """
        Initialize the model wrapper.

        Args:
            params (dict, optional): LightGBM hyperparameters. Defaults to config.LGBM_PARAMS.
        """
        self.params = params if params is not None else LGBM_PARAMS.copy()
        self.model_e = None
        self.model_n = None

    def train(
        self,
        train_df,
        val_df,
        features,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    ):
        """
        Trains the East and North residual models.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame): Validation data containing features and targets.
            features (list): List of feature column names.
            num_boost_round (int): Maximum number of boosting iterations.
            early_stopping_rounds (int): Rounds for early stopping.
        """
        # --- Train East Component ---
        print("\n[LGBMModel] Training East Component Model...")
        X_train = train_df[features]
        y_train_e = train_df["target_E"]
        X_val = val_df[features]
        y_val_e = val_df["target_E"]

        dtrain_e = lgb.Dataset(X_train, label=y_train_e)
        dval_e = lgb.Dataset(X_val, label=y_val_e, reference=dtrain_e)

        self.model_e = lgb.train(
            self.params,
            dtrain_e,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_e, dval_e],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )

        # --- Train North Component ---
        print("\n[LGBMModel] Training North Component Model...")
        y_train_n = train_df["target_N"]
        y_val_n = val_df["target_N"]

        dtrain_n = lgb.Dataset(X_train, label=y_train_n)
        dval_n = lgb.Dataset(X_val, label=y_val_n, reference=dtrain_n)

        self.model_n = lgb.train(
            self.params,
            dtrain_n,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain_n, dval_n],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )

    def predict(self, test_df, features):
        """
        Predicts ENU residuals for the test set.

        Args:
            test_df (pd.DataFrame): Test data containing features.
            features (list): List of feature column names.

        Returns:
            tuple: (pred_e, pred_n) - Numpy arrays of predicted East and North residuals.
        """
        if self.model_e is None or self.model_n is None:
            raise RuntimeError("Models have not been trained yet.")

        X_test = test_df[features]

        # Predict using the best iteration found during training
        pred_e = self.model_e.predict(X_test, num_iteration=self.model_e.best_iteration)
        pred_n = self.model_n.predict(X_test, num_iteration=self.model_n.best_iteration)

        return pred_e, pred_n
