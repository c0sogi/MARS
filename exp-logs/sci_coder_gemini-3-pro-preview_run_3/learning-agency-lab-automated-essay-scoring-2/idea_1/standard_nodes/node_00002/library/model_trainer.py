import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.utils import compute_qwk, post_process_preds


class RidgeRegressor:
    """
    A wrapper for Ridge Regression with built-in Cross-Validation for alpha selection.
    Manages training, evaluation, and persistence of model weights without using pickle.
    """

    def __init__(self):
        """
        Initializes the regressor with hyperparameters from Config.
        """
        self.alphas = Config.RIDGE_ALPHAS
        self.model = None
        self.coef_ = None
        self.intercept_ = None
        self.model_path = os.path.join(Config.MODEL_DIR, "ridge_weights.npz")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Ridge Regression model using RidgeCV to find the best alpha.
        Evaluates on training and validation sets.

        Args:
            X_train (np.ndarray): Training embeddings.
            y_train (np.ndarray): Training target scores.
            X_val (np.ndarray, optional): Validation embeddings.
            y_val (np.ndarray, optional): Validation target scores.
        """
        print("Initializing RidgeCV training...")

        # Initialize RidgeCV
        # cv=None defaults to efficient Leave-One-Out Cross-Validation (LOOCV) for Ridge
        # scoring=None defaults to r2, but for Ridge, optimizing R2 is equivalent to minimizing MSE
        self.model = RidgeCV(
            alphas=self.alphas, scoring="neg_mean_squared_error", store_cv_results=False
        )

        # Fit the model
        print(f"Fitting model on {len(X_train)} samples with alphas: {self.alphas}")
        self.model.fit(X_train, y_train)

        # Extract learned parameters
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        best_alpha = self.model.alpha_

        print(f"Training complete. Best Alpha: {best_alpha}")

        # --- Evaluation ---

        # Training Metrics
        train_preds_raw = self.model.predict(X_train)
        train_preds_int = post_process_preds(train_preds_raw)
        train_qwk = compute_qwk(y_train, train_preds_int)
        train_mse = mean_squared_error(y_train, train_preds_raw)

        print("=== Training Metrics ===")
        print(f"Train MSE: {train_mse}")
        print(f"Train QWK: {train_qwk}")

        # Validation Metrics
        if X_val is not None and y_val is not None:
            val_preds_raw = self.model.predict(X_val)
            val_preds_int = post_process_preds(val_preds_raw)
            val_qwk = compute_qwk(y_val, val_preds_int)
            val_mse = mean_squared_error(y_val, val_preds_raw)

            print("=== Validation Metrics ===")
            print(f"Validation MSE: {val_mse}")
            print(f"Validation QWK: {val_qwk}")

        # Save model weights
        self.save_model()

    def predict(self, X):
        """
        Generates predictions for the given input embeddings.

        Args:
            X (np.ndarray): Input embeddings of shape (n_samples, n_features).

        Returns:
            np.ndarray: Continuous predictions.
        """
        # Ensure model is loaded
        if self.coef_ is None or self.intercept_ is None:
            self.load_model()

        # Linear prediction: y = X @ w + b
        # X shape: (N, D), coef shape: (D,), result: (N,)
        return np.dot(X, self.coef_) + self.intercept_

    def save_model(self):
        """
        Saves the model coefficients and intercept to a .npz file.
        This avoids using pickle.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"Saving model weights to {self.model_path}...")
        np.savez(self.model_path, coef=self.coef_, intercept=self.intercept_)

    def load_model(self):
        """
        Loads the model coefficients and intercept from the .npz file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No saved model found at {self.model_path}. Please train first."
            )

        print(f"Loading model weights from {self.model_path}...")
        data = np.load(self.model_path)
        self.coef_ = data["coef"]
        self.intercept_ = data["intercept"]
