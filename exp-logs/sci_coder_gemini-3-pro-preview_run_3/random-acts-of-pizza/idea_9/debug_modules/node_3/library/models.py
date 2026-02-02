import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from library.config import Config
from library.utils import set_seed


class SparseBagger:
    """
    Level 1 Base Learner: Random Forest Classifier.
    Optimized for high-dimensional sparse inputs (Lexical and Behavioral views).
    """

    def __init__(self, params=None):
        """
        Args:
            params (dict): Hyperparameters for RandomForestClassifier.
        """
        set_seed()
        self.params = params if params else {}
        self.model = RandomForestClassifier(**self.params)
        self.model_name = "SparseBagger"

    def fit(self, X, y):
        """
        Fits the Random Forest model.

        Args:
            X (sparse matrix or array): Feature matrix.
            y (array): Target vector.
        """
        print(f"[{self.model_name}] Training on shape: {X.shape}")
        self.model.fit(X, y)

        # Calculate and print training AUC
        train_probs = self.model.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, train_probs)
        print(f"[{self.model_name}] Training AUC: {train_auc}")
        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.
        """
        return self.model.predict_proba(X)[:, 1]


class DenseBooster:
    """
    Level 1 Base Learner: XGBoost Classifier.
    Optimized for low-dimensional dense inputs (Semantic view).
    Supports early stopping if validation data is provided.
    """

    def __init__(self, params=None):
        """
        Args:
            params (dict): Hyperparameters for XGBClassifier.
        """
        set_seed()
        self.params = params.copy() if params else {}

        # Extract early_stopping_rounds to handle it dynamically in fit()
        # This ensures we don't pass it when no validation set is available (e.g., final retraining)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", None)

        self.model = XGBClassifier(**self.params)
        self.model_name = "DenseBooster"

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fits the XGBoost model.

        Args:
            X (array): Training features.
            y (array): Training targets.
            X_val (array, optional): Validation features for early stopping.
            y_val (array, optional): Validation targets for early stopping.
        """
        print(f"[{self.model_name}] Training on shape: {X.shape}")

        fit_params = {"verbose": False}

        # Configure early stopping only if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            if self.early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model.fit(X, y, **fit_params)

        # Metrics
        train_probs = self.model.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, train_probs)
        print(f"[{self.model_name}] Training AUC: {train_auc}")

        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"[{self.model_name}] Validation AUC: {val_auc}")

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for the positive class.
        """
        return self.model.predict_proba(X)[:, 1]


class StackingMetaLearner:
    """
    Level 2 Meta-Learner: Logistic Regression.
    Combines probability outputs from Level 1 models.
    """

    def __init__(self, params=None):
        """
        Args:
            params (dict): Hyperparameters for LogisticRegression.
        """
        set_seed()
        self.params = params if params else {}
        self.model = LogisticRegression(**self.params)
        self.model_name = "StackingMetaLearner"

    def fit(self, X, y):
        """
        Fits the Logistic Regression meta-learner.

        Args:
            X (array): Matrix of Level 1 predictions (n_samples, n_models).
            y (array): Target vector.
        """
        print(f"[{self.model_name}] Training on shape: {X.shape}")
        self.model.fit(X, y)

        # Metrics
        train_probs = self.model.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, train_probs)
        print(f"[{self.model_name}] Training AUC: {train_auc}")

        # Log coefficients to understand model contribution
        print(f"[{self.model_name}] Coefficients: {self.model.coef_[0]}")
        print(f"[{self.model_name}] Intercept: {self.model.intercept_}")
        return self

    def predict_proba(self, X):
        """
        Predicts final probabilities.
        """
        return self.model.predict_proba(X)[:, 1]
