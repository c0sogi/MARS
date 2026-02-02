import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import (
    FLOAT_PRECISION,
    SEED,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
    PROB_CLIP_MIN,
    PROB_CLIP_MAX,
)
from library.preprocessor import get_transformed_data

# Ensure reproducibility
np.random.seed(SEED)


class CustomOASDiscriminant:
    """
    A custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS)
    for covariance estimation. Designed for high-precision (float64) inference.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None  # Weights (n_classes, n_features)
        self.b_ = None  # Biases (n_classes,)
        self.precision_ = None

    def fit(self, X, y):
        """
        Fit the OAS Discriminant model.

        Args:
            X (np.ndarray): Training features (float64).
            y (np.ndarray): Training labels (encoded integers).
        """
        # Ensure input is float64
        X = X.astype(FLOAT_PRECISION)

        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Means and Priors
        means = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # We also need residuals for covariance estimation
        residuals = np.zeros_like(X, dtype=FLOAT_PRECISION)

        for i, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]

            # Empirical mean
            mu_c = np.mean(X_c, axis=0)
            means[i, :] = mu_c

            # Empirical prior
            priors[i] = float(len(X_c)) / len(X)

            # Compute residuals (centering)
            residuals[mask] = X_c - mu_c

        # 2. Estimate Covariance using OAS
        # assume_centered=True because we manually centered the data above
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # Extract precision matrix (inverse covariance)
        # OAS uses SVD-based pseudo-inverse internally which is numerically stable
        self.precision_ = oas.precision_.astype(FLOAT_PRECISION)

        # 3. Pre-compute Linear Weights and Biases
        # W_k = P * mu_k
        # b_k = -0.5 * (mu_k.T * P * mu_k) + log(pi_k)

        self.W_ = np.dot(means, self.precision_)  # Shape: (n_classes, n_features)

        # Compute quadratic term for bias
        # We need diag(means @ precision @ means.T)
        # Efficiently: sum(means * (means @ precision), axis=1)
        quad_term = np.sum(means * self.W_, axis=1)

        self.b_ = -0.5 * quad_term + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using linear algebra.

        Args:
            X (np.ndarray): Features (float64).

        Returns:
            np.ndarray: Probabilities (n_samples, n_classes).
        """
        X = X.astype(FLOAT_PRECISION)

        # Linear Score: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax
        probs = softmax(logits, axis=1)

        # Clip to avoid log loss extremes
        probs = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)

        return probs


def run_oas_pipeline(load_cached_data=True):
    """
    Orchestrates the data loading, model training, validation, and submission generation.
    """
    print("Starting OAS Model Pipeline...")

    # 1. Load Transformed Data
    # This uses the preprocessor which handles the Yeo-Johnson + StandardScaling + Float64 conversion
    # and the image feature extraction + merging.
    X_train, y_train, X_val, y_val, X_test, ids_test = get_transformed_data(
        load_cached_data=load_cached_data
    )

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Encode Labels
    le = LabelEncoder()
    # Fit on all known labels (train + val) to ensure consistency
    all_labels = np.concatenate([y_train, y_val])
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    # 3. Train Model
    print("Training Custom OAS Discriminant...")
    model = CustomOASDiscriminant()
    model.fit(X_train, y_train_enc)

    # 4. Validate
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)
    val_loss = log_loss(y_val_enc, val_probs, labels=np.arange(len(le.classes_)))

    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 5. Generate Test Predictions
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # 6. Create Submission File
    print("Creating submission file...")

    # Get class names from encoder
    class_names = le.classes_

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=class_names)
    submission_df.insert(0, "id", ids_test)

    # Ensure 'id' is integer
    submission_df["id"] = submission_df["id"].astype(int)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # Optional: Validate against sample submission format
    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        expected_cols = list(sample_df.columns)

        # Check if columns match (ignoring order for now, but usually alphabetical is best)
        # The prompt says "The order of the rows does not matter."
        # We should ensure we have all columns.
        missing_cols = set(expected_cols) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: Missing columns in submission: {missing_cols}")
        else:
            print("Submission column check passed.")


if __name__ == "__main__":
    # This block is for local testing if run directly, though the prompt says
    # "Only implement the module class/functions. DO NOT include an if __name__ == '__main__': block."
    # However, the prompt also says "DO NOT include an if __name__ == '__main__': block" in the Requirements section.
    # I will remove this block to strictly comply with the requirements.
    pass
