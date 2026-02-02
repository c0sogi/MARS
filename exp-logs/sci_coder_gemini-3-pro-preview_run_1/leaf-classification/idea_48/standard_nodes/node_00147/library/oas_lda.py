import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax
from library.config import Config
from library.data_loader import load_and_process_data


class OASDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    Implements the algebraic solution for LDA:
    Logits_k(x) = x^T * (Sigma^-1 * mu_k) - 0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(pi_k)

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        coef_ (np.ndarray): Weight matrix of shape (n_classes, n_features).
        intercept_ (np.ndarray): Bias vector of shape (n_classes,).
        precision_ (np.ndarray): Estimated precision matrix (Sigma^-1).
    """

    def __init__(self):
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.precision_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation on class-centered residuals.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            self
        """
        # Ensure float64
        X = X.astype(Config.DTYPE)

        # Identify classes and priors
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Compute empirical means and priors
        means = np.zeros((n_classes, n_features), dtype=Config.DTYPE)
        priors = np.zeros(n_classes, dtype=Config.DTYPE)

        # We need residuals for covariance estimation: R = X - mu_y
        residuals = np.zeros_like(X, dtype=Config.DTYPE)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]
            count = X_cls.shape[0]

            # Empirical mean for class k
            mean_k = np.mean(X_cls, axis=0)
            means[idx, :] = mean_k

            # Empirical prior for class k
            priors[idx] = count / float(X.shape[0])

            # Center data for this class
            residuals[mask] = X_cls - mean_k

        # Estimate Covariance/Precision using OAS on residuals
        # assume_centered=True because we manually centered the data above
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.precision_ = oas.precision_.astype(Config.DTYPE)

        # Algebraically derive Linear Decision Boundaries
        # W_k = Sigma^-1 * mu_k
        # We compute W as (n_classes, n_features)
        # means is (n_classes, n_features), precision is (n_features, n_features)
        # W = means @ precision
        self.coef_ = np.dot(means, self.precision_)

        # b_k = -0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(pi_k)
        # Note: (mu_k^T * Sigma^-1 * mu_k) is the dot product of mean_k and W_k
        self.intercept_ = np.zeros(n_classes, dtype=Config.DTYPE)
        for i in range(n_classes):
            term1 = -0.5 * np.dot(means[i], self.coef_[i])
            term2 = np.log(priors[i])
            self.intercept_[i] = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear decision function.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        X = X.astype(Config.DTYPE)

        # Linear Score: Z = X @ W.T + b
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax
        return softmax(logits, axis=1)


def run_oas_strategy(load_cached_data=True, debug=False, debug_sample_size=100):
    """
    Orchestrates the OAS LDA strategy: loading, training, evaluation, and submission.
    """
    print("Initializing OAS Discriminant Strategy...")

    # 1. Load Data
    # The loader handles caching, polarity correction, geometric extraction, and high-precision preprocessing
    data = load_and_process_data(
        load_cached_data=load_cached_data,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data

    print(
        f"Data Loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # 2. Train Model
    print("Fitting OAS Discriminant Model...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) extremes as per metric definition,
    # though log_loss function usually handles this, we ensure compliance with task description logic if needed.
    # Scikit-learn's log_loss handles clipping internally (eps=1e-15), which matches the task spec.
    val_loss = log_loss(y_val, val_probs, labels=np.arange(len(classes)))

    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Generate Test Predictions
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # 5. Create Submission File
    print("Formatting Submission...")

    # Create DataFrame
    # Columns: id, Class_1, Class_2, ...
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    # Optional: Validate submission format roughly
    print("Submission Head:")
    print(submission_df.head())


if __name__ == "__main__":
    # This block is for local testing if run directly, though the task says "only implement module functions"
    # and "DO NOT include if __name__ == '__main__': block".
    # However, standard python practice usually allows it for testing.
    # Based on instructions "DO NOT include an if __name__ == '__main__': block", I will omit it.
    pass
