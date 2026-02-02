import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from library import config
from library.data_processor import LeafDataProcessor


class OASPrecisionClassifier:
    """
    A Custom Linear Discriminant Classifier that uses the OAS estimator for
    covariance regularization and utilizes the estimator's internal precision
    matrix (pseudo-inverse) for robust weight calculation.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.coef_ = None
        self.intercept_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        """
        # Enforce high precision
        X = X.astype(np.float64)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.priors_[idx] = X_c.shape[0] / float(n_samples)

        # 2. Compute Residuals (centered data)
        X_centered = X.copy()
        for idx, c in enumerate(self.classes_):
            X_centered[y == c] -= self.means_[idx]

        # 3. Estimate Covariance/Precision using OAS
        # Cite solution_lesson_node_00062: Use library-provided precision_ for robustness
        # in ill-conditioned regimes rather than manually solving the linear system.
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        # Scikit-learn's OAS (via EmpiricalCovariance) computes precision_ using
        # linalg.pinvh (pseudo-inverse) which is more stable than Cholesky for
        # near-singular matrices.
        self.precision_ = oas.precision_.astype(np.float64)

        # 4. Calculate Weights
        # W = Means * Precision (Sigma^-1)
        self.coef_ = np.dot(self.means_, self.precision_)

        # 5. Compute Bias (Intercept)
        self.intercept_ = np.zeros(n_classes, dtype=np.float64)
        for i in range(n_classes):
            # dot product of mean vector and weight vector for class i
            # mahalanobis_term = mu_k^T * Sigma^-1 * mu_k
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
