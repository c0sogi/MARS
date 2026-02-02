import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import seed_everything


class ModelWrapper:
    """
    Abstract base class defining the interface for the Unified Heterogeneous Tri-Ensemble models.
    """

    def __init__(self):
        self.model = None
        seed_everything(Config.SEED)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model with the given data, supporting early stopping.
        """
        raise NotImplementedError("Subclasses must implement fit method.")

    def predict(self, X):
        """
        Predicts probabilities for the positive class (contact).
        """
        raise NotImplementedError("Subclasses must implement predict method.")

    def save(self, path):
        """
        Saves the model wrapper to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """
        Loads the model wrapper from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)


class LGBMExpert(ModelWrapper):
    """
    LightGBM Expert utilizing Leaf-wise growth for dense numerical data.
    """

    def __init__(self):
        super().__init__()
        self.params = Config.LGBM_PARAMS.copy()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Create LightGBM datasets
        train_ds = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_ds]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
            valid_sets.append(val_ds)
            valid_names.append("valid")

        # Train with callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            params=self.params,
            train_set=train_ds,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        # LightGBM predict returns raw probabilities for binary classification
        return self.model.predict(X)


class XGBExpert(ModelWrapper):
    """
    XGBoost Expert utilizing Level-wise growth and dynamic scale_pos_weight.
    """

    def __init__(self):
        super().__init__()
        self.params = Config.XGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Calculate dynamic scale_pos_weight based on training data
        num_pos = np.sum(y_train)
        num_neg = len(y_train) - num_pos
        ratio = float(num_neg) / float(num_pos) if num_pos > 0 else 1.0

        self.params["scale_pos_weight"] = ratio

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        # Sanitize verbose_eval for XGBoost (requires > 0 or False)
        verbose_eval = Config.VERBOSE_EVAL
        if isinstance(verbose_eval, int) and verbose_eval <= 0:
            verbose_eval = False

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=Config.NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=verbose_eval,
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
