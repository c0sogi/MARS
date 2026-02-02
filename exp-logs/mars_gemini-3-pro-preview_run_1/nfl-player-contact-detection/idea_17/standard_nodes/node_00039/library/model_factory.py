import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import save_joblib, load_joblib, calculate_mcc


class LGBMWrapper:
    """
    Wrapper for LightGBM models with support for Scout (balanced) and Expert (imbalanced) modes.
    """

    def __init__(self, mode="scout"):
        """
        Args:
            mode (str): 'scout' or 'expert'.
                        'expert' enables 'is_unbalance=True'.
        """
        self.mode = mode
        self.model = None
        self.params = Config.LGBM_PARAMS.copy()

        if self.mode == "expert":
            self.params["is_unbalance"] = True
            # Ensure metric is set
            if "metric" not in self.params:
                self.params["metric"] = "auc"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation labels.
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        print(f"Training LightGBM ({self.mode} mode)...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=Config.N_ESTIMATORS,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        if X_val is not None and y_val is not None:
            # Evaluate MCC on validation set using best iteration
            preds = self.predict(X_val)
            # Simple 0.5 threshold for logging, actual thresholding happens later
            preds_bin = (preds > 0.5).astype(int)
            mcc = calculate_mcc(y_val, preds_bin)
            print(f"LGBM ({self.mode}) Validation MCC (at 0.5 threshold): {mcc}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # predict returns raw probabilities for binary classification in LGBM
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        """
        Saves the model object.
        """
        save_joblib(self, path)
        print(f"LGBM model saved to {path}")

    @staticmethod
    def load(path):
        """
        Loads a saved model object.
        """
        print(f"Loading LGBM model from {path}...")
        return load_joblib(path)


class XGBWrapper:
    """
    Wrapper for XGBoost models with support for Scout and Expert modes.
    """

    def __init__(self, mode="scout"):
        """
        Args:
            mode (str): 'scout' or 'expert'.
                        'expert' calculates scale_pos_weight dynamically.
        """
        self.mode = mode
        self.model = None
        self.params = Config.XGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.
        """
        # Handle Class Imbalance for Expert Mode
        if self.mode == "expert":
            num_pos = y_train.sum()
            num_neg = len(y_train) - num_pos
            if num_pos > 0:
                scale_pos_weight = num_neg / num_pos
                self.params["scale_pos_weight"] = scale_pos_weight
                print(
                    f"XGBoost Expert Mode: Set scale_pos_weight to {scale_pos_weight}"
                )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        print(f"Training XGBoost ({self.mode} mode)...")

        # Sanitize verbose_eval for XGBoost (requires False or positive int)
        verbose_eval = Config.VERBOSE_EVAL
        if isinstance(verbose_eval, int) and verbose_eval <= 0:
            verbose_eval = False

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=Config.N_ESTIMATORS,
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=verbose_eval,
        )

        if X_val is not None and y_val is not None:
            # Evaluate MCC
            preds = self.predict(X_val)
            preds_bin = (preds > 0.5).astype(int)
            mcc = calculate_mcc(y_val, preds_bin)
            print(f"XGB ({self.mode}) Validation MCC (at 0.5 threshold): {mcc}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self, path):
        """
        Saves the model object.
        """
        save_joblib(self, path)
        print(f"XGB model saved to {path}")

    @staticmethod
    def load(path):
        """
        Loads a saved model object.
        """
        print(f"Loading XGB model from {path}...")
        return load_joblib(path)
