import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from library.config import Config


class XGBoostWrapper:
    """
    Wrapper for XGBoost training and inference tailored for the Heterogeneous Ensemble.
    """

    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        self.fit_params = Config.XGB_FIT_PARAMS.copy()
        self.model = None
        self.le = LabelEncoder()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with Early Stopping and GPU acceleration.
        """
        # Encode targets to 0-indexed range
        y_train_enc = self.le.fit_transform(y_train)
        y_val_enc = self.le.transform(y_val)

        # Update parameters with dynamic info
        self.params["num_class"] = len(self.le.classes_)

        # Create optimized DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_enc)
        dval = xgb.DMatrix(X_val, label=y_val_enc)

        # Train model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.fit_params.get("num_boost_round", 5000),
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.fit_params.get("early_stopping_rounds", 50),
            verbose_eval=self.fit_params.get("verbose_eval", 100),
        )

        # Calculate and print full precision validation accuracy
        # Use best_iteration + 1 because iteration_range is exclusive at the upper bound
        best_limit = self.model.best_iteration + 1
        preds = self.model.predict(dval, iteration_range=(0, best_limit))
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(y_val_enc, pred_labels)
        print(f"XGBoost Final Validation Accuracy: {acc}")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.
        """
        dtest = xgb.DMatrix(X)
        # Ensure we use the best iteration found during training
        if hasattr(self.model, "best_iteration"):
            return self.model.predict(
                dtest, iteration_range=(0, self.model.best_iteration + 1)
            )
        return self.model.predict(dtest)


class LightGBMWrapper:
    """
    Wrapper for LightGBM training and inference tailored for the Heterogeneous Ensemble.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.fit_params = Config.LGBM_FIT_PARAMS.copy()
        self.model = None
        self.le = LabelEncoder()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with Early Stopping and GPU acceleration.
        """
        # Encode targets to 0-indexed range
        y_train_enc = self.le.fit_transform(y_train)
        y_val_enc = self.le.transform(y_val)

        # Update parameters
        self.params["num_class"] = len(self.le.classes_)

        # Create Datasets
        train_data = lgb.Dataset(X_train, label=y_train_enc)
        val_data = lgb.Dataset(X_val, label=y_val_enc, reference=train_data)

        # Configure callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.fit_params.get("early_stopping_rounds", 50)
            ),
            lgb.log_evaluation(period=self.fit_params.get("verbose_eval", 100)),
        ]

        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.fit_params.get("num_boost_round", 5000),
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # Calculate and print full precision validation accuracy
        preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(y_val_enc, pred_labels)
        print(f"LightGBM Final Validation Accuracy: {acc}")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.
        """
        return self.model.predict(X, num_iteration=self.model.best_iteration)
