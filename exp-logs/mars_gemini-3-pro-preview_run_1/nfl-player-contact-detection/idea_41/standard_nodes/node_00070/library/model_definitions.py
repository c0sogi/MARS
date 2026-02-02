import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from library.config import LGBM_PARAMS, XGB_PARAMS, SEED
from library.utils import setup_logger


class LGBMExpert:
    """
    LightGBM Expert Model (Leaf-wise growth).
    Part of the Unified Heterogeneous Dual-Ensemble.
    """

    def __init__(self, params=None, logger=None):
        self.params = params if params else LGBM_PARAMS.copy()
        self.model = None
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "lgbm_expert.log"))
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Trains the LightGBM model.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
            feature_names (list, optional): List of feature names.
        """
        self.logger.info("Initializing LightGBM training...")

        # Create LightGBM Datasets
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names if feature_names else "auto",
        )
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_set,
                feature_name=feature_names if feature_names else "auto",
            )
            valid_sets.append(val_set)
            valid_names.append("valid")

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        # Train
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.params.get("n_estimators", 2000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Log final metric
        if X_val is not None and y_val is not None:
            # Predict to get final metric manually for full precision logging
            preds = self.model.predict(X_val)
            # Calculate LogLoss manually or rely on internal best_score
            best_score = self.model.best_score["valid"]["binary_logloss"]
            self.logger.info(f"LightGBM Best Validation LogLoss: {best_score}")

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def save(self, filepath):
        """Saves the model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"LightGBM model saved to {filepath}")

    def load(self, filepath):
        """Loads the model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.logger.info(f"LightGBM model loaded from {filepath}")


class XGBExpert:
    """
    XGBoost Expert Model (Level-wise growth).
    Part of the Unified Heterogeneous Dual-Ensemble.
    """

    def __init__(self, params=None, logger=None):
        self.params = params if params else XGB_PARAMS.copy()
        self.model = None
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "xgb_expert.log"))
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Trains the XGBoost model.
        Dynamically calculates scale_pos_weight based on training data balance.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
            feature_names (list, optional): List of feature names.
        """
        self.logger.info("Initializing XGBoost training...")

        # Calculate scale_pos_weight dynamically
        # scale_pos_weight = count(negative) / count(positive)
        num_pos = np.sum(y_train)
        num_neg = len(y_train) - num_pos
        ratio = num_neg / num_pos if num_pos > 0 else 1.0

        self.params["scale_pos_weight"] = ratio
        self.logger.info(f"Dynamic scale_pos_weight set to: {ratio}")

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            evals.append((dval, "valid"))

        # Train
        # verbose_eval=False to keep stdout clean, we log manually or rely on final print
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 2000),
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        # Log final metric
        if X_val is not None and y_val is not None:
            # XGBoost stores evaluation history in the object if evals_result is passed,
            # but simpler to just get best score attribute
            best_score = self.model.best_score
            self.logger.info(f"XGBoost Best Validation LogLoss: {best_score}")

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # XGBoost predict requires DMatrix
        # Note: If X is a DataFrame, feature names are preserved.
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, filepath):
        """Saves the model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"XGBoost model saved to {filepath}")

    def load(self, filepath):
        """Loads the model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.logger.info(f"XGBoost model loaded from {filepath}")
