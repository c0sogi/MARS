import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from library.config import (
    LR_SOLVER,
    LR_MAX_ITER,
    LR_C,
    LR_MULTI_CLASS,
    LR_N_JOBS,
    SEED,
    WORKING_DIR,
)
from library.utils import set_seed


class LogRegClassifier:
    """
    Wrapper for Logistic Regression Classifier with custom caching and evaluation logic.
    """

    def __init__(self):
        """
        Initializes the Logistic Regression model with configurations from library.config.
        """
        set_seed(SEED)
        self.model = LogisticRegression(
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            C=LR_C,
            multi_class=LR_MULTI_CLASS,
            n_jobs=LR_N_JOBS,
            random_state=SEED,
            verbose=0,  # Ensure silent execution
        )
        self.model_path = os.path.join(WORKING_DIR, "logreg_model.npz")

    def _save_model(self, path):
        """
        Saves model coefficients and attributes to a .npz file (avoiding pickle).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            classes=self.model.classes_,
        )
        print(f"Model weights saved to {path}")

    def _load_model(self, path):
        """
        Loads model coefficients and attributes from a .npz file and reconstructs the model.
        """
        print(f"Loading model weights from {path}")
        data = np.load(path, allow_pickle=True)

        # Manually set attributes to reconstruct the fitted state
        self.model.classes_ = data["classes"]
        self.model.coef_ = data["coef"]
        self.model.intercept_ = data["intercept"]

        # Set n_features_in_ required for some sklearn checks
        self.model.n_features_in_ = self.model.coef_.shape[1]

        # Mock intercept_scaling (default is 1.0)
        self.model.intercept_scaling = 1.0

    def train(self, X_train, y_train, load_cached_model=True):
        """
        Trains the model or loads it from cache.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            load_cached_model (bool): Whether to try loading a saved model.
        """
        if load_cached_model and os.path.exists(self.model_path):
            try:
                self._load_model(self.model_path)
                return
            except Exception as e:
                print(f"Failed to load cached model: {e}. Retraining...")

        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)

        # Calculate and print training accuracy for sanity check
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"Training Accuracy: {train_acc}")

        self._save_model(self.model_path)

    def evaluate(self, X_val, y_val):
        """
        Evaluates the model on validation data using Log Loss.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            float: The log loss score.
        """
        print("Evaluating model...")
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X_val)

        # Calculate Log Loss
        # labels parameter ensures we handle cases correctly even if a class is missing in val
        loss = log_loss(y_val, y_pred_proba, labels=self.model.classes_)

        print(f"Validation Log Loss: {loss}")
        return loss

    def predict(self, X_test):
        """
        Generates predictions for the test set.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        print("Generating predictions...")
        return self.model.predict_proba(X_test)

    def get_classes(self):
        """
        Returns the list of class names/indices the model was trained on.
        """
        return self.model.classes_
