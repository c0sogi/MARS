import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import RF_PARAMS, WORKING_DIR


class InteractionRandomForest:
    """
    Encapsulates the Stream A Random Forest model logic.
    Utilizes interaction-projected features to capture credibility-consistency dynamics.
    """

    def __init__(self):
        """
        Initializes the RandomForestClassifier with parameters from config.
        """
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.model_path = os.path.join(WORKING_DIR, "rf_model.joblib")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target vector.
            X_val (np.ndarray, optional): Validation feature matrix.
            y_val (np.ndarray, optional): Validation target vector.
        """
        print("Starting Random Forest training...")

        # Fit the model
        self.model.fit(X_train, y_train)

        # Calculate and print training metrics
        train_preds = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_preds)
        print(f"RF Training ROC AUC: {train_auc}")

        # Calculate and print validation metrics if data is provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"RF Validation ROC AUC: {val_auc}")

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class (pizza received).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Array of probabilities for the positive class.
        """
        # predict_proba returns [prob_class_0, prob_class_1]
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        """
        Saves the trained model to the working directory.
        """
        joblib.dump(self.model, self.model_path)
        print(f"RF model saved to {self.model_path}")

    def load(self):
        """
        Loads the trained model from the working directory if it exists.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"RF model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved RF model found at {self.model_path}")
            return False
