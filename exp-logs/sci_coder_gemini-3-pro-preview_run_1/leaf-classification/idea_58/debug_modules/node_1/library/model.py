import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax
import logging
from library.utils import set_seed, save_submission, setup_logging
from library.pipeline import DataPipeline

# Ensure logging is set up
setup_logging()


class OASDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS)
    for covariance estimation. Performs exact analytical inference in float64.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None  # Weight matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)
        self.precision_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation on class residuals.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
        """
        # Ensure float64 for precision
        X = X.astype(np.float64)

        # 1. Identify classes and compute statistics
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # Compute empirical means and priors
        for idx, cls in enumerate(self.classes_):
            X_k = X[y == cls]
            means[idx] = np.mean(X_k, axis=0)
            priors[idx] = X_k.shape[0] / X.shape[0]

        # 2. Compute Residuals (Centered Data)
        # Subtract the corresponding class mean from each sample
        # This creates the pooled within-class scatter basis
        residuals = (
            X - means[y]
        )  # Broadcasting based on label indices if y matches indices
        # Note: y contains encoded labels 0..K-1. If not, we'd need a mapping.
        # The pipeline uses LabelEncoder, so y is 0..98.
        # However, let's be safe and map correctly if classes aren't 0..N
        if not np.array_equal(self.classes_, np.arange(n_classes)):
            # Remap residuals calculation if classes are not sequential integers
            residuals = np.zeros_like(X)
            for idx, cls in enumerate(self.classes_):
                mask = y == cls
                residuals[mask] = X[mask] - means[idx]
        else:
            residuals = X - means[y]

        # 3. Estimate Covariance using OAS
        # assume_centered=True because we explicitly subtracted means
        logging.info("Fitting OAS estimator on residuals...")
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.precision_ = oas.precision_  # The inverse covariance matrix

        # 4. Derive Weights and Bias for Linear Formulation
        # W_k = P * mu_k (Column vector in math, but we store W as (K, D))
        # W = means @ precision (Shape: K x D)
        # Since precision is symmetric: (P mu)^T = mu^T P
        self.W_ = np.dot(means, self.precision_)

        # Bias b_k = -0.5 * (mu_k^T * P * mu_k) + log(pi_k)
        # Quadratic term: diag(means @ W.T)
        quad_term = np.sum(means * self.W_, axis=1)  # Efficient diagonal computation
        self.b_ = -0.5 * quad_term + np.log(priors)

        logging.info(
            f"Model fitted. W shape: {self.W_.shape}, b shape: {self.b_.shape}"
        )
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear discriminant function.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        X = X.astype(np.float64)

        # Linear Discriminant Function: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        probs = softmax(logits, axis=1)
        return probs


def train_and_evaluate(debug=False, load_cached_data=True):
    """
    Runs the full training, evaluation, and submission pipeline.

    Args:
        debug (bool): If True, runs on a subset.
        load_cached_data (bool): If True, loads pre-processed data from cache.
    """
    set_seed(42)

    # 1. Pipeline Execution
    pipeline = DataPipeline(debug=debug, seed=42)
    data = pipeline.run(load_cached_data=load_cached_data)

    X_train, y_train, ids_train = data["train"]
    X_val, y_val, ids_val = data["val"]
    X_test, ids_test = data["test"]
    class_names = data["classes"]

    logging.info(f"Training Data Shape: {X_train.shape}")
    logging.info(f"Validation Data Shape: {X_val.shape}")
    logging.info(f"Test Data Shape: {X_test.shape}")

    # 2. Model Training
    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Validation Evaluation
    logging.info("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Apply metric clipping as per task description for accurate local scoring
    # max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Normalize rows after clipping (though log_loss does this internally usually,
    # the prompt implies rescaling happens before scoring)
    val_probs_clipped /= val_probs_clipped.sum(axis=1, keepdims=True)

    score = log_loss(y_val, val_probs_clipped)
    logging.info(f"Validation Multi-class Log Loss: {score:.15f}")

    # 4. Test Prediction & Submission
    logging.info("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # Ensure correct column mapping
    # The model predicts indices 0..K-1 corresponding to class_names
    # save_submission expects columns in specific order?
    # Usually submission file requires columns to be named by species.
    # The sample_submission format dictates the column order.
    # We should ensure we map our predictions to the correct column names.

    # Load sample submission to get correct column order
    try:
        sample_sub = pd.read_csv("./input/sample_submission.csv")
        sample_cols = sample_sub.columns.tolist()
        # First column is id, rest are species
        target_species_order = sample_cols[1:]

        # Create a DataFrame for our predictions
        pred_df = pd.DataFrame(test_probs, columns=class_names)

        # Reorder columns to match sample submission
        # Fill missing columns with 0 if any (shouldn't be for this dataset)
        for col in target_species_order:
            if col not in pred_df.columns:
                pred_df[col] = 0.0

        pred_df = pred_df[target_species_order]
        final_probs = pred_df.values
        final_class_names = target_species_order

    except Exception as e:
        logging.warning(
            f"Could not align with sample_submission: {e}. Using model class order."
        )
        final_probs = test_probs
        final_class_names = list(class_names)

    save_submission(
        ids_test, final_probs, final_class_names, filename="./submission/submission.csv"
    )


if __name__ == "__main__":
    # Default execution configuration
    train_and_evaluate(debug=False, load_cached_data=True)
