import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import SEED, EPSILON
from library.utils import save_submission
from library.preprocessing import get_preprocessed_data

# Set deterministic seed
np.random.seed(SEED)


class LinearizedOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Linear Discriminant Classifier that uses Oracle Approximating Shrinkage (OAS)
    for robust covariance estimation and pre-compiles the decision boundary into
    a linear transformation to avoid quadratic numerical instability.

    Implements the 'Dual-Precision Gate':
    - Derivation is performed in float64.
    - Inference is performed in float32.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None  # Weights (n_classes, n_features)
        self.b_ = None  # Bias (n_classes,)

    def fit(self, X, y):
        """
        Fits the model.

        1. Computes Class Means and Priors.
        2. Estimates Shared Covariance (Sigma) using OAS on centered residuals.
        3. Derives Linear Weights W = Sigma^-1 * Mu.
        4. Derives Bias b = -0.5 * Mu^T * Sigma^-1 * Mu + log(Prior).
        """
        # Enforce float64 for high-precision derivation
        X_64 = np.array(X, dtype=np.float64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_64.shape[1]

        # Initialize containers
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # Compute Means and Priors
        for i, c in enumerate(self.classes_):
            X_c = X_64[y == c]
            means[i] = np.mean(X_c, axis=0)
            priors[i] = X_c.shape[0] / X_64.shape[0]

        # Compute Centered Residuals for Shared Covariance Estimation
        # LDA assumes a shared covariance matrix across all classes.
        X_residuals = np.zeros_like(X_64)
        for i, c in enumerate(self.classes_):
            X_residuals[y == c] = X_64[y == c] - means[i]

        # Estimate Precision Matrix (Sigma^-1) using OAS
        # assume_centered=True because we manually centered X_residuals
        oas = OAS(assume_centered=True)
        oas.fit(X_residuals)
        precision_matrix = oas.precision_  # Shape (n_features, n_features)

        # --- Linearization Step ---

        # Compute Weight Matrix W (n_classes, n_features)
        # W_k = (Sigma^-1 * mu_k)^T = mu_k^T * Sigma^-1 (since Sigma is symmetric)
        # Matrix operation: W = Means @ Precision
        W_64 = np.dot(means, precision_matrix)

        # Compute Bias Vector b (n_classes,)
        # b_k = -0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(pi_k)
        # The quadratic term is the diagonal of (Means @ Sigma^-1 @ Means^T)
        # Which is equivalent to row-wise dot product of Means and W
        quad_term = np.sum(means * W_64, axis=1)
        b_64 = -0.5 * quad_term + np.log(priors)

        # --- Dual-Precision Gate ---
        # Cast to float32 for inference
        self.W_ = W_64.astype(np.float32)
        self.b_ = b_64.astype(np.float32)

        return self

    def predict_proba(self, X):
        """
        Calculates probabilities using linearized projection.
        Input is cast to float32 to match weights.
        """
        # Cast input to float32
        X_32 = np.array(X, dtype=np.float32)

        # Linear Projection: Z = X @ W.T + b
        logits = np.dot(X_32, self.W_.T) + self.b_

        # Softmax
        return softmax(logits, axis=1)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def train_and_predict(debug_sample_size=None):
    """
    Main pipeline function.

    Args:
        debug_sample_size (int, optional): If set, truncates training data for debugging.
    """
    print("Initializing Linearized OAS Discriminant Pipeline...")

    # 1. Load and Preprocess Data
    # Uses the inductive pipeline defined in library.preprocessing
    X_train, y_train, X_val, y_val, X_test, ids_test, le = get_preprocessed_data(
        load_cached_data=True
    )

    # Debugging option
    if debug_sample_size:
        print(f"DEBUG: Truncating training data to {debug_sample_size} samples.")
        X_train = X_train[:debug_sample_size]
        y_train = y_train[:debug_sample_size]

    # 2. Train Model
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    model = LinearizedOASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Evaluate on Validation Set
    print("Evaluating on Validation Set...")

    # Filter validation set to classes known to the model (Cite debug_lesson_1)
    val_mask = np.isin(y_val, model.classes_)
    X_val_filtered = X_val[val_mask]
    y_val_filtered = y_val[val_mask]

    val_probs = model.predict_proba(X_val_filtered)

    # Clip probabilities for metric calculation (strictly following metric definition)
    val_probs_clipped = np.clip(val_probs, EPSILON, 1.0 - EPSILON)

    val_loss = log_loss(y_val_filtered, val_probs_clipped, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Generate Submission
    print("Generating predictions for Test Set...")
    test_probs_subset = model.predict_proba(X_test)

    # Retrieve original class names
    class_names = le.classes_

    # Project reduced predictions back to full schema if necessary (Cite debug_lesson_2)
    if test_probs_subset.shape[1] != len(class_names):
        print("DEBUG: Projecting subset predictions to full class schema...")
        test_probs = np.zeros(
            (test_probs_subset.shape[0], len(class_names)), dtype=np.float32
        )
        # model.classes_ contains the integer indices of the classes present in training
        test_probs[:, model.classes_] = test_probs_subset
    else:
        test_probs = test_probs_subset

    # Save submission
    save_submission(ids_test, test_probs, class_names)
    print("Pipeline completed successfully.")
