import numpy as np
from sklearn.covariance import OAS
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax

from library.config import PRECISION_TYPE, SEED
from library.utils import set_seed, validate_precision, save_submission
from library.data_loader import load_and_augment_data
from library.preprocessing import preprocess_features


class LinearOASDiscriminant:
    """
    A custom Linear Discriminant Analysis classifier using the Oracle Approximating Shrinkage (OAS)
    estimator for the covariance matrix. Designed for high-precision (float64) inference.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_estimator_ = None
        self.precision_ = None
        self.W_ = None  # Weights (n_classes, n_features)
        self.b_ = None  # Biases (n_classes,)

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
        """
        validate_precision(X, "X_train")

        # 1. Identify Classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 2. Compute Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=PRECISION_TYPE)
        self.priors_ = np.zeros(n_classes, dtype=PRECISION_TYPE)

        # We need residuals for covariance estimation
        # R = X - mu_y
        residuals = np.zeros_like(X, dtype=PRECISION_TYPE)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Arithmetic Mean
            mean_c = np.mean(X_c, axis=0, dtype=PRECISION_TYPE)
            self.means_[idx] = mean_c

            # Prior
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

            # Center data for this class
            residuals[y == c] = X_c - mean_c

        # 3. Estimate Covariance using OAS
        # We assume centered data because we manually computed residuals
        self.covariance_estimator_ = OAS(assume_centered=True)
        self.covariance_estimator_.fit(residuals)

        # Extract Precision Matrix (Inverse Covariance)
        # OAS provides precision_ attribute which is computed via SVD/pseudo-inverse
        self.precision_ = self.covariance_estimator_.precision_.astype(PRECISION_TYPE)
        validate_precision(self.precision_, "Precision Matrix")

        # 4. Derive Linear Decision Boundaries
        # Discriminant function: delta_k(x) = x.T * P * mu_k - 0.5 * mu_k.T * P * mu_k + log(pi_k)
        # Linear form: Z = X * W.T + b
        # Where W_k = P * mu_k
        # And b_k = -0.5 * (mu_k . W_k) + log(pi_k)

        # Compute W (n_classes, n_features)
        # Transpose means to (n_features, n_classes) for matrix multiplication
        # P is (n_features, n_features)
        # W.T = P @ means.T -> W = (P @ means.T).T = means @ P.T = means @ P (since P is symmetric)
        self.W_ = np.dot(self.means_, self.precision_)

        # Compute b (n_classes,)
        # Dot product of each mean vector with its corresponding weight vector
        # np.sum(means * W, axis=1) does row-wise dot product
        quad_term = -0.5 * np.sum(self.means_ * self.W_, axis=1)
        log_priors = np.log(self.priors_)
        self.b_ = quad_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        validate_precision(X, "X_predict")

        # Linear Inference: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        probabilities = softmax(logits, axis=1)

        return probabilities.astype(PRECISION_TYPE)

    def predict(self, X):
        """
        Predicts class labels for X.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def train_and_evaluate(load_cached_data=True):
    """
    Orchestrates the entire training and evaluation pipeline.

    1. Loads and augments data.
    2. Preprocesses data (Yeo-Johnson + Standard Scaler).
    3. Trains the LinearOASDiscriminant.
    4. Evaluates on Validation set.
    5. Generates Submission for Test set.
    """
    set_seed(SEED)

    print("Initializing Training Pipeline...")

    # 1. Load Data
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, class_names = (
        load_and_augment_data(load_cached_data=load_cached_data)
    )

    # 2. Preprocess Data
    # This handles the inductive fitting (fit on train, transform all)
    X_train, X_val, X_test = preprocess_features(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=load_cached_data
    )

    # 3. Initialize and Train Model
    print("\nTraining Linear OAS Discriminant...")
    model = LinearOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Evaluation
    print("\nEvaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Metrics
    # Log Loss
    val_log_loss = log_loss(y_val, val_probs, labels=model.classes_)

    # Accuracy
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Validation Log Loss: {val_log_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # 5. Generate Submission
    print("\nGenerating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # Ensure probabilities are clipped as per competition rules (handled by metric usually,
    # but good practice for submission stability, though softmax guarantees [0,1])
    # The prompt mentions: predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
    # We apply this clipping before saving.
    epsilon = 1e-15
    test_probs = np.clip(test_probs, epsilon, 1 - epsilon)

    # Save submission
    save_submission(test_ids, test_probs, class_names)

    return val_log_loss, val_acc
