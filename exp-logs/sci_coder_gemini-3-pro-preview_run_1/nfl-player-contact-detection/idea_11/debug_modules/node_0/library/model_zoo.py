import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import WORKING_DIR, N_JOBS, RANDOM_STATE
from library.utils import compute_mcc


class LGBMWrapper:
    def __init__(self, params, name="lgbm"):
        """
        Wrapper for LightGBM Booster.
        """
        self.params = params.copy()
        self.name = name
        self.model = None
        self.best_threshold = 0.5

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
    ):
        """
        Trains the LightGBM model with early stopping.
        """
        # Create LightGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Setup callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]

        # Train
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Evaluate MCC on validation set to report full precision metric
        preds_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        best_mcc = -1.0
        best_th = 0.5
        # Scan thresholds
        thresholds = np.linspace(0.05, 0.95, 91)
        for th in thresholds:
            pred_labels = (preds_val >= th).astype(int)
            mcc = compute_mcc(y_val, pred_labels)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        self.best_threshold = best_th
        print(
            f"[{self.name}] Best Validation MCC: {best_mcc:.16f} at Threshold: {best_th:.4f}"
        )

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        """
        Saves the model to a text file.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path):
        """
        Loads the model from a text file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = lgb.Booster(model_file=path)


class XGBWrapper:
    def __init__(self, params, name="xgb"):
        """
        Wrapper for XGBoost Booster.
        """
        self.params = params.copy()
        self.name = name
        self.model = None
        self.best_threshold = 0.5

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100,
    ):
        """
        Trains the XGBoost model with early stopping.
        """
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        # Evaluate MCC on validation set
        preds_val = self.model.predict(
            dval, iteration_range=(0, self.model.best_iteration + 1)
        )

        best_mcc = -1.0
        best_th = 0.5
        thresholds = np.linspace(0.05, 0.95, 91)
        for th in thresholds:
            pred_labels = (preds_val >= th).astype(int)
            mcc = compute_mcc(y_val, pred_labels)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        self.best_threshold = best_th
        print(
            f"[{self.name}] Best Validation MCC: {best_mcc:.16f} at Threshold: {best_th:.4f}"
        )

    def predict(self, X):
        """
        Predicts probabilities. Converts input to DMatrix automatically.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Handle input types
        if not isinstance(X, xgb.DMatrix):
            dtest = xgb.DMatrix(X)
        else:
            dtest = X

        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self, path):
        """
        Saves the model to a JSON file.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path):
        """
        Loads the model from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = xgb.Booster()
        self.model.load_model(path)


class HeterogeneousEnsemble:
    def __init__(self, models):
        """
        Manages an ensemble of heterogeneous models (LGBM, XGB).

        Args:
            models (list): List of instantiated and trained wrapper objects.
        """
        self.models = models

    def predict(self, X):
        """
        Returns the unweighted average of probabilities from all models.
        """
        if not self.models:
            raise ValueError("No models in ensemble.")

        preds_list = []
        for m in self.models:
            preds_list.append(m.predict(X))

        # Average probabilities
        return np.mean(preds_list, axis=0)

    def optimize_threshold(self, X_val, y_val):
        """
        Finds the optimal decision threshold for the ensemble prediction on validation data.

        Returns:
            float: The optimal threshold.
        """
        preds = self.predict(X_val)

        best_mcc = -1.0
        best_th = 0.5
        thresholds = np.linspace(0.05, 0.95, 91)

        for th in thresholds:
            pred_labels = (preds >= th).astype(int)
            mcc = compute_mcc(y_val, pred_labels)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        print(
            f"[Ensemble] Best Validation MCC: {best_mcc:.16f} at Threshold: {best_th:.4f}"
        )
        return best_th
