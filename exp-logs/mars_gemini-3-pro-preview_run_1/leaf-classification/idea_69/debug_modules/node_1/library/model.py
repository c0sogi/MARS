import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from scipy.special import softmax
from library.utils import set_seed, compute_metric
from library.data_loader import load_dataset


class SanitizedOASDiscriminant:
    """
    Custom Linear Discriminant classifier using the OAS Covariance Backbone.
    Designed for exact analytical inference using float64 precision.
    """

    def __init__(self):
        set_seed(42)
        self.classes_ = None
        self.W_ = None  # Weights matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)

    def fit(self, X, y):
        """
        Fits the model using OAS estimation on residuals.

        Args:
            X (array-like): Training features (n_samples, n_features).
            y (array-like): Training labels (n_samples,).
        """
        # Ensure high precision
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Means and Priors
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # Map class labels to 0..K-1 indices for internal calculation
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for cls in self.classes_:
            idx = class_to_idx[cls]
            X_c = X[y == cls]
            means[idx] = np.mean(X_c, axis=0)
            priors[idx] = len(X_c) / len(X)

        # 2. Compute Residuals (Centering)
        # R = X - mu_y
        # Create a matrix of means corresponding to each sample's class
        y_indices = np.array([class_to_idx[yi] for yi in y])
        R = X - means[y_indices]

        # 3. Estimate Precision Matrix via OAS
        # assume_centered=True because R is explicitly centered above
        estimator = OAS(assume_centered=True)
        estimator.fit(R)
        P = estimator.precision_  # Shape: (n_features, n_features)

        # 4. Derive Linear Weights and Bias
        # W_k = P * mu_k
        # We want W to be (n_classes, n_features) so Z = X @ W.T + b
        # W = (P @ means.T).T = means @ P.T = means @ P (Symmetric P)
        self.W_ = np.matmul(means, P)

        # b_k = -0.5 * (W_k . mu_k) + log(pi_k)
        # Element-wise multiplication followed by sum across features
        term1 = -0.5 * np.sum(self.W_ * means, axis=1)
        term2 = np.log(priors)
        self.b_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear formulation.

        Args:
            X (array-like): Test features (n_samples, n_features).

        Returns:
            array-like: Probabilities (n_samples, n_classes).
        """
        X = np.array(X, dtype=np.float64)

        # Linear Scoring: Z = X @ W.T + b
        logits = np.matmul(X, self.W_.T) + self.b_

        # Probability: Softmax
        probs = softmax(logits, axis=1)

        return probs


def train_and_predict(max_samples=None):
    """
    Executes the training and prediction pipeline.

    Args:
        max_samples (int, optional): Number of samples to use for debugging.
    """
    # 1. Load Data
    # The data_loader handles merging geometric features, sanitization, and scaling.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=True, max_samples=max_samples
    )

    # 2. Initialize and Fit Model
    model = SanitizedOASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Validation
    val_probs = model.predict_proba(X_val)
    # y_val are integer indices from LabelEncoder in data_loader
    val_loss = compute_metric(y_val, val_probs, labels=list(range(len(classes))))
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Test Prediction
    test_probs = model.predict_proba(X_test)

    # 5. Generate Submission
    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Create DataFrame
    df_sub = pd.DataFrame(test_probs, columns=classes)
    df_sub.insert(0, "id", test_ids)

    # Save
    submission_path = os.path.join(submission_dir, "submission.csv")
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
