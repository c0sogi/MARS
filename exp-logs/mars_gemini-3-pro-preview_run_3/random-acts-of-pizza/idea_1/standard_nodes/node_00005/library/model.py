import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class PizzaSuccessModel:
    """
    Wrapper class for the XGBoost model.
    """

    def __init__(self):
        """
        Initializes the XGBoost Classifier with parameters from Config.
        """
        self.params = Config.XGB_PARAMS
        self.clf = XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the XGBoost model and evaluates on the validation set using early stopping.

        Args:
            X_train: Sparse matrix or array of training features.
            y_train: Array of training labels.
            X_val: Sparse matrix or array of validation features (optional).
            y_val: Array of validation labels (optional).
        """
        print("Training XGBoost Classifier...")

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = 100

        self.clf.fit(X_train, y_train, **fit_params)

        # Calculate and print Training AUC
        train_probs = self.clf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        print(f"Training AUC: {train_auc}")

        # Calculate and print Validation AUC if validation data is provided
        if X_val is not None and y_val is not None:
            val_probs = self.clf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"Validation AUC: {val_auc}")

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class.

        Args:
            X: Sparse matrix or array of features.

        Returns:
            Array of probabilities for class 1 (received pizza).
        """
        # predict_proba returns [prob_class_0, prob_class_1]
        return self.clf.predict_proba(X)[:, 1]
