import numpy as np
from scipy.special import softmax
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from library.config import FLOAT_PRECISION, SEED, PROB_CLIP_EPS
from library.utils import set_seed, save_submission

# Set global seed
set_seed(SEED)


class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    Implements the 'Sanitized Hybrid-Geometry High-Precision OAS Discriminant' strategy.
    It computes the linear decision boundaries algebraically using the precision matrix
    derived from the OAS covariance estimator on class-centered residuals.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        priors_ (np.ndarray): Empirical class priors.
        means_ (np.ndarray): Class centroids.
        covariance_estimator (OAS): The fitted sklearn OAS estimator.
        coef_ (np.ndarray): Weight matrix (W) of shape (n_classes, n_features).
        intercept_ (np.ndarray): Bias vector (b) of shape (n_classes,).
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_estimator = None
        self.coef_ = None
        self.intercept_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).

        Returns:
            self
        """
        # Enforce high precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        y = np.array(y)

        # Encode labels to integers for internal processing
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # We calculate means using the arithmetic mean
        for k in range(n_classes):
            X_k = X[y_enc == k]
            self.means_[k] = np.mean(X_k, axis=0)
            self.priors_[k] = len(X_k) / len(X)

        # 2. Compute Centered Residuals
        # R = X - mu_y
        # We subtract the corresponding class mean from each sample
        X_centered = X - self.means_[y_enc]

        # 3. Estimate Covariance using OAS
        # assume_centered=True because we have manually centered the data
        self.covariance_estimator = OAS(assume_centered=True)
        self.covariance_estimator.fit(X_centered)

        # 4. Derive Linear Decision Boundaries
        # Precision Matrix P = Sigma^-1
        P = self.covariance_estimator.precision_.astype(FLOAT_PRECISION)

        # Weights W_k = P * mu_k
        # Since P is symmetric, W = means @ P
        # Shape: (n_classes, n_features)
        self.coef_ = np.dot(self.means_, P)

        # Bias b_k = -0.5 * (mu_k^T * P * mu_k) + log(pi_k)
        #          = -0.5 * (W_k . mu_k) + log(pi_k)
        # We compute the dot product for each class
        # np.sum(self.coef_ * self.means_, axis=1) performs the row-wise dot product
        term1 = -0.5 * np.sum(self.coef_ * self.means_, axis=1)
        term2 = np.log(self.priors_)
        self.intercept_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Args:
            X (array-like): Input features of shape (n_samples, n_features).

        Returns:
            probs (np.ndarray): Probabilities of shape (n_samples, n_classes).
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.array(X, dtype=FLOAT_PRECISION)

        # Linear Scoring: Z = X @ W^T + b
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Probability: Softmax(Z)
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Args:
            X (array-like): Input features.

        Returns:
            preds (np.ndarray): Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """
    Trains the OASLinearDiscriminant model and evaluates it on the validation set.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        model: The trained OASLinearDiscriminant instance.
    """
    print("Initializing OASLinearDiscriminant...")
    model = OASLinearDiscriminant()

    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    print("Evaluating on validation data...")
    # Predict probabilities
    val_probs = model.predict_proba(X_val)

    # Clip probabilities for metric calculation stability (as per task description)
    # max(min(p, 1-10^-15), 10^-15)
    val_probs_clipped = np.clip(val_probs, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)

    # Calculate Log Loss
    # Note: sklearn log_loss handles string labels if we provide the `labels` argument
    loss = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    print(f"Validation Multi-class Log Loss: {loss}")

    return model


def generate_submission(model, X_test, test_ids):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained OASLinearDiscriminant model.
        X_test: Test features.
        test_ids: Test image IDs.
    """
    print("Generating predictions for test set...")
    probs = model.predict_proba(X_test)

    # Ensure probabilities are in [0, 1] and handle clipping for the submission file
    # The task description says: "predicted probabilities are replaced with max(min(p,1-10^{-15}),10^{-15})"
    # We apply this clipping before saving to ensure the submission matches the metric logic exactly.
    probs = np.clip(probs, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)

    # Save submission
    save_submission(test_ids, model.classes_, probs)
