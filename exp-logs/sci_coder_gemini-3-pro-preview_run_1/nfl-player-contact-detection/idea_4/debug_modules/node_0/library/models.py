import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from library.config import Config


class LGBMWrapper:
    """
    Wrapper for LightGBM model with specific handling for the contact detection task.
    Uses 'is_unbalance=True' to handle class imbalance internally.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.seed = Config.SEED

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation labels.
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get("n_estimators", 1000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

        # Print final metric
        if X_val is not None and y_val is not None:
            # Predict on validation to show final score explicitly if needed,
            # though lgb.train logs it.
            pass

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.array: Probabilities of class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, filename):
        """Saves the model to the working directory."""
        path = os.path.join(Config.WORKING_DIR, filename)
        joblib.dump(self.model, path)
        print(f"LGBM model saved to {path}")

    def load(self, filename):
        """Loads the model from the working directory."""
        path = os.path.join(Config.WORKING_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"LGBM model loaded from {path}")
        else:
            print(f"LGBM model file not found at {path}")


class XGBWrapper:
    """
    Wrapper for XGBoost model.
    Dynamically calculates 'scale_pos_weight' to handle class imbalance.
    """

    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        self.model = None
        self.seed = Config.SEED

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation labels.
        """
        # Calculate scale_pos_weight dynamically
        # scale_pos_weight = count(negative) / count(positive)
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        self.params["scale_pos_weight"] = scale_pos_weight
        print(f"XGBoost scale_pos_weight calculated: {scale_pos_weight}")

        # Create DMatrices
        # enable_categorical=True if we had categorical types, but features are mostly float32 now.
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 1000),
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.array: Probabilities of class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self, filename):
        """Saves the model to the working directory."""
        path = os.path.join(Config.WORKING_DIR, filename)
        joblib.dump(self.model, path)
        print(f"XGB model saved to {path}")

    def load(self, filename):
        """Loads the model from the working directory."""
        path = os.path.join(Config.WORKING_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"XGB model loaded from {path}")
        else:
            print(f"XGB model file not found at {path}")
