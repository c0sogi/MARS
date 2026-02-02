import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from abc import ABC, abstractmethod
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import SEED, N_JOBS


class ModelWrapper(ABC):
    """
    Abstract Base Class for unified model interface.
    """

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model with early stopping.

        Args:
            X: Training features.
            y: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.

        Args:
            X: Features to predict on.

        Returns:
            np.ndarray: Probability of class 1.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to disk.

        Args:
            path (str): File path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """
        Loads the model from disk.

        Args:
            path (str): File path to load the model from.

        Returns:
            ModelWrapper: The loaded model instance.
        """
        pass


class LGBMClassifierWrapper(ModelWrapper):
    def __init__(self, params):
        self.params = params.copy()
        self.model = None
        # Ensure verbose is set to -1 to suppress warnings
        self.params["verbosity"] = -1
        self.params["random_state"] = SEED
        self.params["n_jobs"] = N_JOBS

    def fit(self, X, y, X_val=None, y_val=None):
        # Create LightGBM datasets
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")

        # Extract training arguments from params that are not model hyperparameters
        num_boost_round = self.params.pop("n_estimators", 1000)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Train with callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress printing per iteration
        ]

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Print final metrics for verification
        if X_val is not None and y_val is not None:
            preds = self.model.predict(X_val)
            loss = log_loss(y_val, preds)
            # Simple threshold search for reporting
            best_mcc = -1
            for thresh in np.linspace(0.1, 0.9, 9):
                mcc = matthews_corrcoef(y_val, (preds > thresh).astype(int))
                if mcc > best_mcc:
                    best_mcc = mcc
            print(f"[LGBM] Best Validation LogLoss: {loss}")
            print(f"[LGBM] Best Validation MCC (approx): {best_mcc}")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class XGBClassifierWrapper(ModelWrapper):
    def __init__(self, params):
        self.params = params.copy()
        self.model = None
        self.params["random_state"] = SEED
        self.params["n_jobs"] = N_JOBS
        # Ensure verbosity is off
        self.params["verbosity"] = 0

    def fit(self, X, y, X_val=None, y_val=None):
        # Dynamic Class Weighting Calculation
        num_pos = np.sum(y)
        num_neg = len(y) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        self.params["scale_pos_weight"] = scale_pos_weight

        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        # Extract training args
        num_boost_round = self.params.pop("n_estimators", 1000)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        # Print final metrics
        if X_val is not None and y_val is not None:
            # Predict using the best iteration
            preds = self.model.predict(
                dval, iteration_range=(0, self.model.best_iteration + 1)
            )
            loss = log_loss(y_val, preds)
            best_mcc = -1
            for thresh in np.linspace(0.1, 0.9, 9):
                mcc = matthews_corrcoef(y_val, (preds > thresh).astype(int))
                if mcc > best_mcc:
                    best_mcc = mcc
            print(f"[XGB] Best Validation LogLoss: {loss}")
            print(f"[XGB] Best Validation MCC (approx): {best_mcc}")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # XGBoost predict requires DMatrix
        # Check if X is already DMatrix, if not convert
        if not isinstance(X, xgb.DMatrix):
            dtest = xgb.DMatrix(X)
        else:
            dtest = X

        # Predict using best iteration if available
        if hasattr(self.model, "best_iteration"):
            return self.model.predict(
                dtest, iteration_range=(0, self.model.best_iteration + 1)
            )
        else:
            return self.model.predict(dtest)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
