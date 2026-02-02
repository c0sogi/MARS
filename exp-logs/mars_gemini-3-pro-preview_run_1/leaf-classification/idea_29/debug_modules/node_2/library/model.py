import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from library import config
from library.data_processor import LeafDataProcessor


class CholeskyOASClassifier:
    """
    A Custom Linear Discriminant Classifier that uses the OAS estimator for
    covariance regularization and Cholesky decomposition for exact,
    high-precision inversion.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Enforce high precision
        X = X.astype(np.float64)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # We calculate means and priors for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.priors_[idx] = X_c.shape[0] / float(n_samples)

        # 2. Compute Residuals (centered data)
        # LDA assumes a shared covariance matrix calculated from pooled residuals
        X_centered = X.copy()
        for idx, c in enumerate(self.classes_):
            X_centered[y == c] -= self.means_[idx]

        # 3. Estimate Covariance Matrix using OAS
        # OAS guarantees a well-conditioned, positive-definite matrix
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        self.covariance_ = oas.covariance_.astype(np.float64)

        # 4. Solve for Weights using Cholesky Decomposition
        # We want to compute W = Means * Sigma^-1
        # Equivalent to solving Sigma * W^T = Means^T for W^T

        # Perform Cholesky factorization: Sigma = L * L.T
        # lower=True returns lower triangular L
        c, lower = linalg.cho_factor(self.covariance_, lower=True)

        # Solve A * x = b where A=Sigma, b=Means^T, x=W^T
        means_T = self.means_.T
        W_T = linalg.cho_solve((c, lower), means_T)

        self.coef_ = W_T.T  # Shape: (n_classes, n_features)

        # 5. Compute Bias (Intercept)
        # b_k = -0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(prior_k)
        # Note that (mu_k^T * Sigma^-1 * mu_k) is the diagonal of (Means * W^T)
        # But we can compute it efficiently as row-wise dot product of Means and W

        self.intercept_ = np.zeros(n_classes, dtype=np.float64)
        for i in range(n_classes):
            # dot product of mean vector and weight vector for class i
            mahalanobis_term = np.dot(self.means_[i], self.coef_[i])
            self.intercept_[i] = -0.5 * mahalanobis_term + np.log(self.priors_[i])

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        """
        X = X.astype(np.float64)
        # Linear Discriminant Function: Z = X * W^T + b
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax to get probabilities
        return special.softmax(scores, axis=1)


def run_task(load_cached_data=True, debug=False):
    """
    Main execution function to load data, train the Cholesky-OAS model,
    validate, and generate submission.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug (bool): If True, subsets data for quick testing.
    """
    # 1. Load Data
    processor = LeafDataProcessor()
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = processor.load_data(
        load_cached_data=load_cached_data
    )

    if debug:
        print("Debug mode: Subsetting data...")
        X_train = X_train[:100]
        y_train = y_train[:100]

        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows
        # Ensure validation set only contains classes present in the training subset
        train_classes = np.unique(y_train)
        X_val_subset = X_val[:50]
        y_val_subset = y_val[:50]
        mask = np.isin(y_val_subset, train_classes)
        X_val = X_val_subset[mask]
        y_val = y_val_subset[mask]

    # 2. Train Model
    print("Initializing and fitting CholeskyOASClassifier...")
    clf = CholeskyOASClassifier()
    clf.fit(X_train, y_train)

    # 3. Validation
    print("Predicting on validation set...")
    val_probs = clf.predict_proba(X_val)

    # Clip probabilities to avoid log(0) extremes, matching competition metric logic
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)

    # Cite debug_lesson_3: Synchronize Metadata Schema with Data Encoding
    val_loss = log_loss(y_val, val_probs_clipped, labels=clf.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Test Prediction & Submission
    print("Predicting on test set...")
    test_probs = clf.predict_proba(X_test)

    # Prepare submission DataFrame
    # Cite debug_lesson_2: Maintain Global Schema Consistency When Filtering Classes
    if test_probs.shape[1] < len(classes):
        full_probs = np.zeros((test_probs.shape[0], len(classes)), dtype=np.float64)
        full_probs[:, clf.classes_] = test_probs
        test_probs = full_probs

    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save submission
    print(f"Saving submission to {config.SUBMISSION_FILE}...")
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")
