import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import Config
from library.preprocessor import load_and_preprocess_data


class IntegralInertialDiscriminant:
    """
    A High-Precision Linear Discriminant Classifier using OAS Covariance Estimation.

    This classifier implements the 'Integral-Inertial' strategy by fusing
    geometric features with a robust statistical estimator (OAS). It operates
    entirely in float64 to prevent metric floor collapse.
    """

    def __init__(self):
        self.classes_ = None
        self.W = None  # Weights matrix: Shape (n_classes, n_features)
        self.b = None  # Bias vector: Shape (n_classes,)
        self.precision_ = None  # Precision matrix (Inverse Covariance)

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        1. Compute Class Centroids and Priors.
        2. Compute Residuals (X - Centroid).
        3. Estimate Precision Matrix using OAS on Residuals.
        4. Derive Linear Decision Boundaries (W, b).
        """
        # Enforce high precision
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.classes_ = le.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Empirical Means and Priors
        means = np.zeros((n_classes, n_features), dtype=Config.FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=Config.FLOAT_PRECISION)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / n_samples

        # 2. Compute Residuals
        # Subtract the corresponding class mean from each sample
        # This centers the data around the origin relative to its class
        X_centered = X - means[y_encoded]

        # 3. Estimate Covariance/Precision via OAS
        # We assume centered data because we manually calculated intra-class residuals.
        # This pools the covariance estimate across all classes (LDA assumption).
        estimator = OAS(assume_centered=True)
        estimator.fit(X_centered)

        self.precision_ = estimator.precision_.astype(Config.FLOAT_PRECISION)

        # 4. Derive Linear Weights and Biases
        # W_k = Sigma^-1 * mu_k = P * mu_k
        # We compute W as (n_classes, n_features)
        # means.T is (n_features, n_classes)
        # precision_ is (n_features, n_features)
        # W.T = P @ means.T -> W = (P @ means.T).T
        self.W = np.dot(self.precision_, means.T).T

        # b_k = -0.5 * (mu_k^T * P * mu_k) + log(pi_k)
        # Note: mu_k^T * P * mu_k is equivalent to mu_k . W_k (dot product)
        # We compute this for all classes at once using element-wise mult and sum
        quadratic_term = np.sum(means * self.W, axis=1)
        self.b = -0.5 * quadratic_term + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Probability estimation using Softmax on Linear Logits.
        """
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Linear Projection: Z = X @ W.T + b
        logits = np.dot(X, self.W.T) + self.b

        # Apply Softmax in high precision
        probs = softmax(logits, axis=1)

        return probs


def train_and_evaluate(load_cached_data=True):
    """
    Orchestrates the training, validation, and submission generation process.
    """
    print("Loading and preprocessing data...")
    # Load data using the provided library function
    # This handles feature extraction, cleaning, transformation, and caching
    data = load_and_preprocess_data(load_cached_data=load_cached_data)
    X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test = data

    print(f"Training IntegralInertialDiscriminant on {len(X_train)} samples...")
    model = IntegralInertialDiscriminant()
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # labels=model.classes_ ensures the columns of val_probs match the target encoding
    score = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {score}")

    print("Generating predictions for test set...")
    test_probs = model.predict_proba(X_test)

    # Create Submission DataFrame
    # Columns must be id, Class1, Class2, ...
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", ids_test)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return score
