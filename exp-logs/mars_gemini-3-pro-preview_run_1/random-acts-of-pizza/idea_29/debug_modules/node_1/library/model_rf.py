import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import RF_PARAMS, CACHE_DIR


class PizzaRandomForest:
    """
    Stream A: Scope-Restricted Sentiment-Aware Random Forest.

    This model utilizes metadata, engineered ratios, sentiment scores, and high-fidelity TF-IDF features.
    It explicitly excludes sparse history features to maintain orthogonal scoping relative to the neural network.
    """

    def __init__(self):
        """
        Initialize the Random Forest model with parameters from config.
        """
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.model_path = os.path.join(CACHE_DIR, "rf_model.joblib")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest classifier and evaluates on validation data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series/np.array): Training labels.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series/np.array, optional): Validation labels.

        Returns:
            float: Validation ROC AUC score if validation data is provided, else None.
        """
        print("Starting Random Forest training...")

        # Fit the model
        self.model.fit(X_train, y_train)

        # Training metrics
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        print(f"RF Training ROC AUC: {train_auc}")

        # Validation metrics
        val_auc = None
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            # Print full precision as requested
            print(f"RF Validation ROC AUC: {val_auc}")

        return val_auc

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.array: Probabilities of class 1 (received pizza).
        """
        # Return probabilities for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        """
        Saves the trained model to the cache directory.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"RF model saved to {self.model_path}")

    def load(self):
        """
        Loads the trained model from the cache directory if it exists.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"RF model loaded from {self.model_path}")
            return True
        return False
