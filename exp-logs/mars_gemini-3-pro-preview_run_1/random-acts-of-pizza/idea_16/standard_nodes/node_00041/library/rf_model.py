import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config


class LatentSemanticRF:
    """
    Stream A: Latent Semantic Random Forest.
    Wraps a RandomForestClassifier designed to consume the concatenated
    TF-IDF + LSA + Metadata feature matrix.
    """

    def __init__(self, params=None):
        """
        Initialize the model with parameters.

        Args:
            params (dict, optional): Hyperparameters for RandomForestClassifier.
                                     Defaults to config.RF_PARAMS.
        """
        self.params = params if params is not None else config.RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.model_path = os.path.join(config.WORKING_DIR, "rf_model.joblib")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model and evaluates on validation data if provided.
        Saves the trained model to disk.

        Args:
            X_train (sparse matrix or array): Training features.
            y_train (array): Training targets.
            X_val (sparse matrix or array, optional): Validation features.
            y_val (array, optional): Validation targets.
        """
        print(
            f"Training LatentSemanticRF with {X_train.shape[0]} samples and {X_train.shape[1]} features..."
        )

        # Fit the model
        self.model.fit(X_train, y_train)

        # Evaluate on Training Data
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        print(f"RF Train AUC: {train_auc}")

        # Evaluate on Validation Data if provided
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            # Requirement: Print full precision
            print(f"RF Validation AUC: {val_auc}")

        # Save the model
        self.save()

    def predict(self, X):
        """
        Generates probability predictions for the positive class (pizza received).

        Args:
            X (sparse matrix or array): Features to predict on.

        Returns:
            np.array: Probabilities of class 1.
        """
        # Check if model is fitted
        if not hasattr(self.model, "estimators_"):
            # Try to load if not in memory
            if os.path.exists(self.model_path):
                print(f"Loading RF model from {self.model_path}...")
                self.model = joblib.load(self.model_path)
            else:
                raise RuntimeError("Model is not fitted and no saved model found.")

        return self.model.predict_proba(X)[:, 1]

    def save(self):
        """
        Saves the current model to the working directory.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"RF model saved to {self.model_path}")

    def load(self):
        """
        Loads the model from the working directory.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"RF model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")
