import os
import numpy as np
import cv2
from sklearn.linear_model import Ridge
from library.config import (
    PATCH_SIZE,
    ALPHA,
    MODEL_WEIGHTS_PATH,
    MODEL_BIAS_PATH,
    WORKING_DIR,
)


class LearnedLinearFilter:
    """
    A patch-based linear regressor for image denoising.
    Learns a convolutional kernel (weights) and bias to map noisy patches to clean pixels.
    """

    def __init__(self, patch_size=PATCH_SIZE, alpha=ALPHA):
        self.patch_size = patch_size
        self.alpha = alpha
        # Initialize Ridge Regression model
        # solver='auto' efficiently handles the linear system
        self.model = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
        self.kernel = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the linear filter on provided patch data and saves the learned weights.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, patch_size*patch_size).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        print("Training LearnedLinearFilter (Ridge Regression)...")

        # Fit the model
        self.model.fit(X, y)

        # Extract learned parameters
        # Reshape coefficients to (k, k) kernel
        # We cast to float32 for compatibility with cv2.filter2D
        self.kernel = self.model.coef_.reshape(self.patch_size, self.patch_size).astype(
            np.float32
        )
        self.bias = float(self.model.intercept_)

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Save parameters for caching/inference
        np.save(MODEL_WEIGHTS_PATH, self.kernel)
        np.save(MODEL_BIAS_PATH, np.array([self.bias]))
        print(f"Model weights saved to {WORKING_DIR}")

        # Calculate and print training metric
        train_score = self.model.score(X, y)
        print("Training R^2 score:")
        print(train_score)

    def load_weights(self):
        """
        Loads learned weights from the cache directory.

        Returns:
            bool: True if weights were successfully loaded, False otherwise.
        """
        if os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists(MODEL_BIAS_PATH):
            try:
                self.kernel = np.load(MODEL_WEIGHTS_PATH)
                self.bias = float(np.load(MODEL_BIAS_PATH))
                return True
            except Exception as e:
                print(f"Error loading weights: {e}")
                return False
        return False

    def predict(self, image):
        """
        Applies the learned filter to a full image to remove noise.

        Args:
            image (np.ndarray): Normalized input image (H, W) with values in [0, 1].

        Returns:
            np.ndarray: Denoised image (H, W) with values clipped to [0, 1].
        """
        # Ensure model weights are available
        if self.kernel is None:
            if not self.load_weights():
                raise RuntimeError(
                    "Model is not fitted and no cached weights found. Call fit() first."
                )

        # Apply the learned linear kernel using 2D convolution (correlation)
        # borderType=cv2.BORDER_REFLECT handles edges by reflecting pixels,
        # which is consistent with the training patch extraction logic.
        # ddepth=-1 maintains the input depth (float).
        denoised = cv2.filter2D(image, -1, self.kernel, borderType=cv2.BORDER_REFLECT)

        # Add the learned bias (intercept)
        denoised = denoised + self.bias

        # Clip values to valid grayscale range [0, 1]
        denoised = np.clip(denoised, 0.0, 1.0)

        return denoised

    def evaluate(self, X, y):
        """
        Evaluates the model on a validation set of patches and prints the RMSE.

        Args:
            X (np.ndarray): Validation feature matrix.
            y (np.ndarray): Validation target vector.

        Returns:
            float: The Root Mean Squared Error on the validation set.
        """
        if self.kernel is None and not self.load_weights():
            raise RuntimeError("Model not fitted.")

        # Predict on patch features using the underlying sklearn model
        # (Re-constructing the linear equation: y = Xw + b)
        # Note: We can use self.model.predict(X) if fit() was called in this session.
        # If loaded from disk, self.model might not be fitted, so we compute manually.

        # Flatten kernel for dot product
        w_flat = self.kernel.flatten()
        preds = np.dot(X, w_flat) + self.bias

        mse = np.mean((y - preds) ** 2)
        rmse = np.sqrt(mse)

        print("Validation RMSE (Patch-level):")
        print(rmse)

        return rmse
