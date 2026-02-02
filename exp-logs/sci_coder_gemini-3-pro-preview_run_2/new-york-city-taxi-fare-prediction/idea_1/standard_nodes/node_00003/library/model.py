import os
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from library.config import Config
from library.utils import calculate_rmse


class PhysicsInformedLinearModel:
    """
    A linear regression model optimized for the NYC Taxi Fare prediction task.
    Wraps sklearn.linear_model.SGDRegressor with physics-informed constraints
    and custom training logic (early stopping).
    """

    def __init__(self):
        # Load hyperparameters from Config
        self.params = Config.get_model_params()

        # Initialize the underlying SGDRegressor
        # We handle early stopping manually if a validation set is provided
        self.model = SGDRegressor(**self.params)

        self.min_fare = Config.MIN_FARE
        self.batch_size = Config.BATCH_SIZE

    def fit(self, X_train, y_train, X_val=None, y_val=None, patience=5):
        """
        Trains the model using Stochastic Gradient Descent.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray, optional): Validation features for early stopping.
            y_val (np.ndarray, optional): Validation targets for early stopping.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        # If validation set is not provided, fallback to standard fit
        if X_val is None or y_val is None:
            print(
                "No validation set provided. Training without explicit early stopping."
            )
            self.model.fit(X_train, y_train)
            return

        # Manual training loop with early stopping
        max_iter = self.params.get("max_iter", 1000)
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))

        best_rmse = float("inf")
        no_improve_epochs = 0
        best_weights = None
        best_intercept = None

        print(
            f"Starting training: {n_samples} samples, {n_batches} batches/epoch, max_iter={max_iter}"
        )

        for epoch in range(max_iter):
            # Shuffle training data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.model.partial_fit(X_batch, y_batch)

            # Validation step
            val_preds = self.predict(X_val)
            val_rmse = calculate_rmse(y_val, val_preds)

            # Print full precision as requested
            print(f"Epoch {epoch + 1}: Validation RMSE = {val_rmse}")

            # Early Stopping Logic
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_weights = self.model.coef_.copy()
                best_intercept = self.model.intercept_.copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered. Best RMSE: {best_rmse}")
                break

        # Restore best model weights
        if best_weights is not None:
            self.model.coef_ = best_weights
            self.model.intercept_ = best_intercept

    def predict(self, X):
        """
        Predicts fare amounts for the given features.
        Enforces the minimum fare constraint.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        try:
            predictions = self.model.predict(X)
        except NotFittedError:
            # Fallback calculation if sklearn attributes aren't fully set after loading
            predictions = np.dot(X, self.model.coef_.T) + self.model.intercept_

        # Enforce lower bound (Physics/Domain constraint)
        return np.maximum(predictions, self.min_fare)

    def save(self, path):
        """
        Saves the model weights to a file.

        Args:
            path (str): File path to save the model (e.g., 'model.npz').
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Using np.savez to avoid pickle and save only necessary parameters
        np.savez(path, coef=self.model.coef_, intercept=self.model.intercept_)
        print(f"Model weights saved to {path}")

    def load(self, path):
        """
        Loads model weights from a file.

        Args:
            path (str): File path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path)
        self.model.coef_ = data["coef"]
        self.model.intercept_ = data["intercept"]

        # Manually set attributes to bypass NotFittedError in standard sklearn usage
        # This acts as a flag that the model is fitted
        self.model.t_ = 1
        self.model.n_features_in_ = self.model.coef_.shape[0]

        print(f"Model weights loaded from {path}")
