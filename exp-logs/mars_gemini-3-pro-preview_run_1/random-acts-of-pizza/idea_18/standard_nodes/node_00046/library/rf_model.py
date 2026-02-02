import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed


class RFModel:
    def __init__(self):
        """
        Initializes the Random Forest Model wrapper with hyperparameters from Config.
        """
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            class_weight=Config.RF_CLASS_WEIGHT,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_STATE,
            verbose=0,
        )
        self.model_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES["rf_model"]
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest classifier.

        Args:
            X_train: Training features (sparse matrix or array).
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
        """
        set_seed(Config.RANDOM_STATE)

        print("Training Random Forest...")
        self.model.fit(X_train, y_train)

        # Training Metrics
        train_preds = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_preds)
        print(f"Train AUC: {train_auc}")

        # Validation Metrics
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Validation AUC: {val_auc}")

        self.save()

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class.

        Args:
            X: Features to predict on.

        Returns:
            np.array: Probabilities of class 1.
        """
        # Ensure model is loaded or trained
        if not hasattr(self.model, "estimators_"):
            self.load()

        # Predict
        probs = self.model.predict_proba(X)

        # Return probability of the positive class (index 1)
        if probs.shape[1] == 2:
            return probs[:, 1]
        else:
            # Handle edge case where model might only see one class (unlikely with proper data)
            return probs[:, 0] if self.model.classes_[0] == 1 else np.zeros(len(X))

    def save(self):
        """
        Saves the trained model to disk using joblib.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"RF Model saved to {self.model_path}")

    def load(self):
        """
        Loads the trained model from disk.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"RF Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"RF Model not found at {self.model_path}. Please train first."
            )
