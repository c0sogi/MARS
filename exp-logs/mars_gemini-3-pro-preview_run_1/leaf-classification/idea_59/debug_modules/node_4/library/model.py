import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax

from library import config, data_handler, preprocessor


class OASLinearModel(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimation.

    Implements the 'Sanitized Robust-Integral High-Precision OAS Discriminant' strategy.
    Analytically derives linear decision boundaries using the precision matrix
    from the Oracle Approximating Shrinkage estimator.
    """

    def __init__(self):
        self.classes_ = None
        self.le = None
        self.W = None  # Weights matrix (n_classes, n_features)
        self.b = None  # Bias vector (n_classes,)
        self.precision_ = None
        self.shrinkage_ = None

    def fit(self, X, y):
        """
        Fits the model using analytical linear discriminant analysis with OAS shrinkage.

        Args:
            X (array-like): Feature matrix (n_samples, n_features).
            y (array-like): Target labels (n_samples,).
        """
        # Enforce high precision
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Encode labels
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_

        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Compute empirical class means and priors
        means = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        for k in range(n_classes):
            X_k = X[y_enc == k]
            if len(X_k) > 0:
                means[k] = np.mean(X_k, axis=0)
                priors[k] = len(X_k) / n_samples
            else:
                # Fallback for empty classes (should not happen with stratified split)
                priors[k] = 1.0 / n_samples

        # Compute Residuals (Centering)
        # We subtract the class mean from each sample to estimate the common covariance
        X_centered = np.zeros_like(X, dtype=config.FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_enc == k
            if np.any(mask):
                X_centered[mask] = X[mask] - means[k]

        # Estimate Covariance using OAS
        # assume_centered=True because we have manually removed the class means
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        self.precision_ = oas.precision_.astype(config.FLOAT_PRECISION)
        self.shrinkage_ = oas.shrinkage_

        # Derive Linear Weights and Biases
        # Weight vector w_k = P * mu_k
        # W matrix (rows are w_k^T): W = (P @ means.T).T = means @ P (since P is symmetric)
        self.W = np.dot(means, self.precision_)

        # Bias scalar b_k = -0.5 * (mu_k^T P mu_k) + log(pi_k)
        #                 = -0.5 * (mu_k . w_k) + log(pi_k)
        # We compute the dot product row-wise
        quadratic_term = -0.5 * np.sum(means * self.W, axis=1)

        # Add log priors (add epsilon to avoid log(0) just in case)
        log_priors = np.log(priors + 1e-15)

        self.b = quadratic_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear decision function.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Linear Inference: Z = X W^T + b
        # X: (N, D), W: (K, D), b: (K,)
        logits = np.dot(X, self.W.T) + self.b

        # Apply Softmax
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_and_inference(debug_size=None, load_cached_data=True):
    """
    Orchestrates the full pipeline:
    1. Loads data (Train, Val, Test).
    2. Preprocesses data (Sanitization -> Transformation).
    3. Trains OASLinearModel.
    4. Evaluates on Validation set.
    5. Generates Submission for Test set.
    """
    print("Initializing Pipeline...")

    # 1. Load Data
    X_train_raw, y_train, train_ids = data_handler.load_dataset(
        "train", debug_size=debug_size, load_cached_data=load_cached_data
    )
    X_val_raw, y_val, val_ids = data_handler.load_dataset(
        "val", debug_size=debug_size, load_cached_data=load_cached_data
    )
    X_test_raw, _, test_ids = data_handler.load_dataset(
        "test", debug_size=debug_size, load_cached_data=load_cached_data
    )

    # 2. Preprocess Data
    # Determine suffix for caching based on debug size
    debug_suffix = f"_debug_{debug_size}" if debug_size is not None else ""

    X_train, X_val, X_test = preprocessor.get_transformed_data(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        debug_suffix=debug_suffix,
        load_cached_data=load_cached_data,
    )

    # 3. Train Model
    print("Training OASLinearModel...")
    model = OASLinearModel()
    model.fit(X_train, y_train)

    print(f"Model Fitted. OAS Shrinkage: {model.shrinkage_:.6f}")

    # 4. Evaluation
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # sklearn log_loss supports string labels if labels=model.classes_ is passed.
    score = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {score:.15f}")

    # 5. Submission Generation
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # Clip probabilities as per task requirement
    # max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    test_probs = np.clip(test_probs, epsilon, 1 - epsilon)

    # Create Submission DataFrame
    # Columns must be id, then species in alphabetical order.
    # model.classes_ comes from LabelEncoder, which sorts alphabetically.
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save
    print(f"Saving submission to {config.OUTPUT_SUBMISSION_PATH}...")
    submission_df.to_csv(config.OUTPUT_SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
