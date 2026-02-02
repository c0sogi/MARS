import os
import copy
import warnings
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from library.config import Config
from library.utils import compute_log_loss


class AuthorClassifier:
    """
    A wrapper class for the Logistic Regression model with support for
    Early Stopping and model persistence.
    """

    def __init__(self):
        """
        Initializes the AuthorClassifier with parameters from Config.
        """
        # Create a copy of parameters to avoid modifying the global Config
        self.params = Config.MODEL_PARAMS.copy()

        # Extract max_iter to control the outer Early Stopping loop
        # Default to 1000 if not specified in Config
        self.max_iter = self.params.pop("max_iter", 1000)

        # Initialize LogisticRegression with warm_start=True.
        # We set max_iter=1 for the internal solver to allow step-by-step training.
        # This enables us to check validation loss after each 'epoch'.
        self.model = LogisticRegression(**self.params, warm_start=True, max_iter=1)

        self.is_fitted = False
        self.classes_ = None

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        patience=5,
        load_cached_model=True,
    ):
        """
        Trains the model using the provided training data.
        Implements Early Stopping based on validation loss if validation data is provided.

        Args:
            X_train (sparse matrix): Training features.
            y_train (array-like): Training labels.
            X_val (sparse matrix, optional): Validation features.
            y_val (array-like, optional): Validation labels.
            patience (int): Number of epochs to wait for improvement before stopping.
            load_cached_model (bool): If True, attempts to load a saved model from disk.
        """
        # Define path for caching the model
        model_cache_path = os.path.join(Config.WORKING_DIR, "model.joblib")

        # 1. Attempt to load from cache
        if load_cached_model:
            if os.path.exists(model_cache_path):
                print(f"Loading cached model from {model_cache_path}...")
                try:
                    cached_data = joblib.load(model_cache_path)
                    self.model = cached_data["model"]
                    self.classes_ = self.model.classes_
                    self.is_fitted = True
                    print("Model loaded successfully.")
                    return
                except Exception as e:
                    print(
                        f"Failed to load cached model: {e}. Proceeding to train from scratch."
                    )

        # 2. Train from scratch
        print("Starting training with Early Stopping...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Suppress ConvergenceWarning as we manually control iterations with max_iter=1
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        best_loss = float("inf")
        no_improvement_count = 0
        best_model_state = None

        # Check if validation data is available for Early Stopping
        has_val = (X_val is not None) and (y_val is not None)

        for epoch in range(self.max_iter):
            # Perform one epoch of training
            self.model.fit(X_train, y_train)

            # Store classes if not already stored
            if self.classes_ is None:
                self.classes_ = self.model.classes_

            if has_val:
                # Predict on validation set
                val_probs = self.model.predict_proba(X_val)

                # Calculate loss
                # Note: We assume model.classes_ matches Config.CLASSES (alphabetical order)
                current_loss = compute_log_loss(y_val, val_probs)

                # Print full precision as requested
                print(
                    f"Epoch {epoch + 1}/{self.max_iter} - Validation Log Loss: {current_loss}"
                )

                # Check for improvement
                if current_loss < best_loss:
                    best_loss = current_loss
                    no_improvement_count = 0
                    # Deep copy the underlying model to save the best state
                    best_model_state = copy.deepcopy(self.model)
                else:
                    no_improvement_count += 1

                # Early Stopping check
                if no_improvement_count >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. Best Validation Loss: {best_loss}"
                    )
                    if best_model_state is not None:
                        self.model = best_model_state
                    break
            else:
                # If no validation set, just print progress occasionally
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.max_iter} complete.")

        self.is_fitted = True

        # 3. Save model to cache
        print(f"Saving model to {model_cache_path}...")
        try:
            joblib.dump({"model": self.model}, model_cache_path)
        except Exception as e:
            print(f"Warning: Failed to save model to cache: {e}")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input features.

        Args:
            X (sparse matrix): Input features.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        # Scikit-learn LogisticRegression sorts classes alphabetically.
        # Config.CLASSES = ['EAP', 'HPL', 'MWS'] is also alphabetical.
        # So columns correspond correctly to the target classes.
        return self.model.predict_proba(X)
