import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax

from library.config import FLOAT_PRECISION, SUBMISSION_DIR, SAMPLE_SUBMISSION_PATH
from library.data import load_data


class SpectralSpatialOAS:
    """
    A custom Linear Discriminant Classifier that uses Oracle Approximating Shrinkage (OAS)
    for robust covariance estimation in high-dimensional spectral-spatial feature spaces.

    It computes the linear decision boundaries analytically and performs inference
    strictly in float64 precision to avoid numerical instability.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_estimator = OAS(assume_centered=True)
        self.W_ = None  # Weights matrix
        self.b_ = None  # Bias vector

    def fit(self, X, y):
        """
        Fits the model by estimating class means, priors, and the shared covariance matrix.

        Args:
            X (np.ndarray): Training features (float64).
            y (np.ndarray): Training labels (encoded integers).
        """
        # Ensure float64
        X = X.astype(FLOAT_PRECISION)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.priors_[idx] = float(len(X_c)) / n_samples

        # 2. Compute Centered Residuals
        # We subtract the class mean corresponding to each sample's label
        # to get the intra-class variations.
        X_centered = X - self.means_[y]

        # 3. Estimate Covariance using OAS
        # We use assume_centered=True because we just centered the data manually.
        self.covariance_estimator.fit(X_centered)

        # 4. Derive Linear Decision Boundaries
        # Precision matrix P = inverse(Sigma)
        precision = self.covariance_estimator.precision_.astype(FLOAT_PRECISION)

        # Weights: W_k = P * mu_k
        # Shape: (n_classes, n_features)
        # We transpose means to (n_features, n_classes) for dot product, then transpose back
        self.W_ = np.dot(self.means_, precision)

        # Bias: b_k = -0.5 * (mu_k^T * W_k) + log(pi_k)
        # We compute the quadratic term efficiently
        # np.sum(self.means_ * self.W_, axis=1) computes the dot product for each class row
        quadratic_term = -0.5 * np.sum(self.means_ * self.W_, axis=1)
        log_priors = np.log(self.priors_)

        self.b_ = quadratic_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear scoring function.

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        X = X.astype(FLOAT_PRECISION)

        # Linear Score: Z = X * W^T + b
        # X: (n_samples, n_features)
        # W_: (n_classes, n_features) -> W_.T: (n_features, n_classes)
        # b_: (n_classes,)
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        proba = softmax(logits, axis=1)

        return proba.astype(FLOAT_PRECISION)

    def predict(self, X):
        """
        Predicts class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def train_and_predict(load_cached_data=True, max_samples=None):
    """
    Main execution function to load data, train the SpectralSpatialOAS model,
    evaluate performance, and generate the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        max_samples (int, optional): Limit dataset size for debugging.
    """
    # 1. Load Data
    # The load_data function handles feature extraction, merging, and high-precision preprocessing
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        load_cached_data=load_cached_data, max_samples=max_samples
    )

    # 2. Train Model
    print("Initializing Spectral-Spatial OAS Discriminant...")
    model = SpectralSpatialOAS()

    print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features...")
    model.fit(X_train, y_train)

    # 3. Evaluate on Validation Set
    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Calculate metrics
    # Note: log_loss requires probabilities
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Validation Log Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # 4. Generate Test Predictions
    print("Generating predictions for test set...")
    test_probs = model.predict_proba(X_test)

    # 5. Create Submission File
    # The submission format requires 'id' and columns for each species
    submission_df = pd.DataFrame(test_probs, columns=classes)

    # Align with full schema to handle debug subsets (Cite debug_lesson_2)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH, nrows=1)
    required_species = [c for c in sample_sub.columns if c != "id"]
    submission_df = submission_df.reindex(columns=required_species, fill_value=0.0)

    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    print(f"Saving submission to {submission_path}...")
    submission_df.to_csv(submission_path, index=False)
    print("Done.")
