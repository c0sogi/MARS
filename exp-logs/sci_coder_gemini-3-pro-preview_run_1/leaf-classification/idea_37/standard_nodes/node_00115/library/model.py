import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax

from library import config, data_loader, preprocessing


class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Analysis using Oracle Approximating Shrinkage (OAS)
    for covariance estimation. Designed for high-precision float64 inference.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        means_ (np.ndarray): Class means matrix (n_classes, n_features).
        priors_ (np.ndarray): Class priors vector (n_classes,).
        precision_ (np.ndarray): Precision matrix (inverse covariance) (n_features, n_features).
        coef_ (np.ndarray): Weight matrix W (n_classes, n_features).
        intercept_ (np.ndarray): Bias vector b (n_classes,).
    """

    def __init__(self, assume_centered=True):
        self.assume_centered = assume_centered
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation on class-conditional residuals.

        Args:
            X (np.ndarray): Training data (n_samples, n_features), float64.
            y (np.ndarray): Target labels (n_samples,), int.
        """
        # Ensure float64
        X = X.astype(config.FLOAT_PRECISION)

        # 1. Parameter Estimation
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        # Compute means and priors
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i, :] = np.mean(X_c, axis=0)
            self.priors_[i] = float(len(X_c)) / len(y)

        # 2. Compute Residuals (Centering)
        # Subtract the corresponding class mean from each sample
        # We map y to indices in self.classes_ to index into self.means_
        # Since y is already label encoded 0..N-1 by data_loader, we can use it directly if sorted
        # But to be safe, we map explicitly.
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        indices = np.array([class_to_idx[yi] for yi in y])
        means_expanded = self.means_[indices]
        residuals = X - means_expanded

        # 3. Estimate Covariance via OAS
        # We use assume_centered=True because we manually centered the data
        estimator = OAS(assume_centered=self.assume_centered)
        estimator.fit(residuals)

        # Extract precision matrix (SVD-based pseudo-inverse handled by sklearn)
        self.precision_ = estimator.precision_.astype(config.FLOAT_PRECISION)

        # 4. Derive Linear Decision Boundaries
        # W = M @ P
        # shape: (n_classes, n_features) = (n_classes, n_features) @ (n_features, n_features)
        self.coef_ = np.dot(self.means_, self.precision_)

        # b = -0.5 * diag(M @ P @ M.T) + log(priors)
        # Efficiently: -0.5 * sum(W * M, axis=1) + log(priors)
        # W * M is element-wise multiplication. Sum over features gives the quadratic term.
        quadratic_term = np.sum(self.coef_ * self.means_, axis=1)
        self.intercept_ = -0.5 * quadratic_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized decision function.

        Args:
            X (np.ndarray): Input data (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        X = X.astype(config.FLOAT_PRECISION)

        # Linear Scoring: Z = X @ W.T + b
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax in float64
        return softmax(logits, axis=1)


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training and prediction pipeline.

    1. Loads and fuses data (Tabular + Visual).
    2. Applies High-Precision Preprocessing (Yeo-Johnson + Scaling).
    3. Trains the Augmented OAS Discriminant.
    4. Evaluates on Validation Set.
    5. Generates Submission for Test Set.
    """
    print("Initializing Augmented High-Precision OAS Discriminant Pipeline...")

    # 1. Load Data (Fusing Tabular and Visual Features)
    # Returns DataFrames and numpy arrays
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, class_names = (
        data_loader.load_and_augment_data(load_cached_data=load_cached_data)
    )

    # 2. Preprocessing (Yeo-Johnson + Standard Scaling in float64)
    # Returns numpy arrays
    X_train, X_val, X_test = preprocessing.process_and_cache_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=load_cached_data
    )

    print(
        f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 3. Model Training
    print("Training OAS Linear Discriminant...")
    model = OASLinearDiscriminant(assume_centered=config.OAS_ASSUME_CENTERED)
    model.fit(X_train, y_train)

    # 4. Validation
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) extremes, though metric function usually handles it.
    # We use the raw probabilities for the metric call as sklearn handles it.
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)

    print("-" * 40)
    print(f"Validation Multi-class Log Loss: {val_loss}")
    print("-" * 40)

    # 5. Submission Generation
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    # Create Submission DataFrame
    # Columns: id, class_1, class_2, ...
    # The order of columns in test_probs corresponds to model.classes_ (0, 1, 2...)
    # which corresponds to class_names sorted by LabelEncoder.

    submission_df = pd.DataFrame(test_probs, columns=class_names)
    submission_df.insert(0, "id", test_ids)

    # Save Submission
    print(f"Saving submission to {config.SUBMISSION_FILE_PATH}...")
    submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    # This block is for local testing only, the competition runner imports functions.
    run_training_pipeline(load_cached_data=True)
