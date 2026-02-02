import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import warnings

# Import from provided library
from library.config import SEED, SUBMISSION_DIR, SAMPLE_SUBMISSION_PATH
from library.utils import set_seed
from library.data import load_data
from library.model import SpectralSpatialOAS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization
    set_seed(SEED)

    # 2. Load Data
    # We load the full dataset to ensure the best possible covariance estimation for OAS.
    # The analytical nature of the model ensures this is fast (< 1 min).
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        load_cached_data=True, max_samples=None
    )

    # 3. Model Training
    print("Training Spectral-Spatial OAS Discriminant...")
    model = SpectralSpatialOAS()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation...")
    # Predict probabilities
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # y_val contains integer indices [0, K-1]. model.classes_ are these indices.
    metric = log_loss(y_val, val_probs, labels=model.classes_)

    # Print metric in strict format
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("Running failure analysis...")
    # Calculate error magnitude: 1.0 - Probability assigned to the true class
    # y_val is an array of integer class indices
    rows = np.arange(len(y_val))
    true_class_probs = val_probs[rows, y_val]
    error_magnitude = 1.0 - true_class_probs

    # Calculate correlation between features and error magnitude
    feature_correlations = []
    n_features = X_val.shape[1]

    for i in range(n_features):
        feature_vec = X_val[:, i]
        # Handle constant features to avoid division by zero in correlation
        if np.std(feature_vec) > 1e-12:
            corr = np.corrcoef(feature_vec, error_magnitude)[0, 1]
            feature_correlations.append((i, corr))
        else:
            feature_correlations.append((i, 0.0))

    # Sort by absolute correlation strength (descending)
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for idx, corr in feature_correlations[:5]:
        print(f"Feature Index {idx}: Correlation {corr:.4f}")

    # 6. Submission Logic
    # Threshold defined in task
    THRESHOLD = 5.234670549314967e-14

    if metric < THRESHOLD:
        print(
            f"Metric {metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate Test Predictions
        test_probs = model.predict_proba(X_test)

        # Create DataFrame
        # classes array from load_data contains the string species names sorted alphabetically
        submission_df = pd.DataFrame(test_probs, columns=classes)

        # Add ID column
        submission_df.insert(0, "id", test_ids)

        # Ensure alignment with sample submission format
        # Load sample submission header to guarantee correct column ordering
        if os.path.exists(SAMPLE_SUBMISSION_PATH):
            sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH, nrows=1)
            # Reindex to match sample columns (this handles order and any missing columns if applicable)
            submission_df = submission_df.reindex(
                columns=sample_sub.columns, fill_value=0.0
            )

        # Save Submission
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Metric {metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
