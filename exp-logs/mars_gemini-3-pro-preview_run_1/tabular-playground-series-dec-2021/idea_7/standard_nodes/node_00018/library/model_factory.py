import xgboost as xgb
import numpy as np
import pandas as pd


class XGBWrapper:
    def __init__(self, params, num_boost_round, early_stopping_rounds):
        """
        A wrapper class for XGBoost training and inference, handling DMatrix conversion,
        label adjustment, and early stopping.

        Args:
            params (dict): Parameters for the XGBoost model.
            num_boost_round (int): Maximum number of boosting rounds.
            early_stopping_rounds (int): Rounds of no improvement to stop training.
        """
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training targets (expected 1-based class labels).
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation targets.
        """
        # XGBoost requires 0-based indexing for multi-class classification.
        # The dataset provides 1-based labels (1 to 7), so we subtract 1.
        y_train_adj = y_train - 1
        dtrain = xgb.DMatrix(data=X_train, label=y_train_adj)

        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            y_val_adj = y_val - 1
            dval = xgb.DMatrix(data=X_val, label=y_val_adj)
            evals.append((dval, "val"))

        # Train the model
        # verbose_eval=False to suppress massive log output.
        # We print the best score manually at the end.
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        # Print the best score with full precision as required
        if hasattr(self.model, "best_score"):
            print(f"Training completed. Best Score: {self.model.best_score}")
            print(f"Best Iteration: {self.model.best_iteration}")

    def predict_proba(self, X):
        """
        Generates class probabilities for the input data.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction.")

        dtest = xgb.DMatrix(data=X)

        # Use the best iteration found during training for prediction
        # iteration_range expects (start, end)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )
