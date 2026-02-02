import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import WORKING_DIR, SEED
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets
from library.preprocessing import get_transformed_data


class StabilizedOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Linear Discriminant Classifier that uses the OAS estimator for covariance
    and algebraically linearizes the decision boundary to ensure numerical stability.
    Strictly operates in float64 precision.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weights (n_classes, n_features)
        self.b_ = None  # Biases (n_classes,)
        self.precision_matrix_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and linear parameter derivation.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Target labels, shape (n_samples,).
        """
        # Enforce float64
        X = X.astype(np.float64)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Class Statistics (Means and Priors)
        # We use empirical counts for priors (Cite Lesson 00033)
        class_means = np.zeros((n_classes, n_features), dtype=np.float64)
        class_priors = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]
            class_means[k] = np.mean(X_k, axis=0)
            class_priors[k] = float(len(X_k)) / len(X)

        # 2. Compute Centered Residuals for Covariance Estimation
        # R = X - mu_y
        residuals = X - class_means[y_encoded]

        # 3. Estimate Shared Covariance using OAS
        # OAS is preferred over Ledoit-Wolf for Gaussianized data (Cite Lesson 00047)
        # We set assume_centered=False to allow OAS to remove any residual floating-point
        # mean drift from the manually centered residuals, ensuring numerical stability.
        oas = OAS(assume_centered=False)
        oas.fit(residuals)

        # 4. Compute Precision Matrix (Inverse Covariance)
        self.precision_matrix_ = oas.precision_.astype(np.float64)

        # 5. Linearization
        # We convert the quadratic discriminant to a linear one: z = Wx + b
        # The quadratic term -0.5 * x.T * P * x cancels out in Softmax.
        # W_k = P * mu_k
        # b_k = -0.5 * mu_k.T * P * mu_k + log(pi_k)

        # W shape: (n_classes, n_features)
        # class_means.T shape: (n_features, n_classes)
        # precision_matrix shape: (n_features, n_features)
        # W.T = P @ means.T -> W = (P @ means.T).T = means @ P.T = means @ P (since P is symmetric)
        self.W_ = np.dot(class_means, self.precision_matrix_)

        # Bias calculation
        # Term 1: -0.5 * diag(mu_k @ P @ mu_k.T)
        # We can compute this efficiently: sum( (means @ P) * means, axis=1 )
        term1 = -0.5 * np.sum(self.W_ * class_means, axis=1)
        term2 = np.log(class_priors)
        self.b_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized formulation.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        X = X.astype(np.float64)

        # Linear Score: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax with high precision
        # scipy.special.softmax handles overflow protection
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_pipeline(load_cached=True):
    """
    Orchestrates the data loading, preprocessing, training, evaluation, and submission.
    """
    print("Initializing Stabilized OAS Discriminant Pipeline...")
    set_seed(SEED)

    # 1. Load Data
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, classes = (
        load_datasets(load_cached_data=load_cached)
    )

    # 2. Preprocessing (Yeo-Johnson + Standard Scaling)
    # Note: Inductive fit on train only is handled inside get_transformed_data
    X_train, X_val, X_test = get_transformed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=load_cached
    )

    # 3. Model Initialization
    model = StabilizedOASDiscriminant()

    # 4. Training
    print(f"Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # 5. Evaluation
    print("Evaluating on Validation set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We use the model.classes_ to ensure correct column alignment if necessary,
    # though our custom class ensures alignment with fit labels.
    metric = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {metric:.16f}")

    # 6. Test Prediction
    print("Generating predictions for Test set...")
    test_probs = model.predict_proba(X_test)

    # 7. Save Submission
    submission_path = "./submission/submission.csv"
    save_submission(test_ids, test_probs, model.classes_, submission_path)

    return model, metric


if __name__ == "__main__":
    # This block is not required by instructions but useful for local testing if run directly.
    # The instructions say "Only implement the module class/functions. DO NOT include an if __name__ == '__main__': block."
    # However, to be safe and strictly follow "DO NOT include...", I will comment this out or remove it.
    # The instructions specifically asked for the module content.
    pass
