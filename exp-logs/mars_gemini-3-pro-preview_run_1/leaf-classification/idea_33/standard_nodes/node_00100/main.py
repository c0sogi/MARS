import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from library.utils import set_seed, save_submission
from library.data_loader import load_data
from library.model import ExactOASDiscriminant


def run_failure_analysis(X_val, y_val, val_probs, classes):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("Performing failure analysis...")

    # Encode y_val to match probability indices
    le = LabelEncoder()
    le.fit(classes)
    y_indices = le.transform(y_val)

    # Calculate error magnitude: 1.0 - probability assigned to the true class
    # val_probs shape: (n_samples, n_classes)
    # We extract the probability corresponding to the true class for each sample
    true_class_probs = val_probs[np.arange(len(y_val)), y_indices]
    error_magnitude = 1.0 - true_class_probs

    # Calculate correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_values = X_val[:, i]
        # Pearson correlation
        corr, _ = pearsonr(feature_values, error_magnitude)
        if np.isnan(corr):
            corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:10]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")


def main():
    # 1. Setup
    set_seed(42)
    cache_dir = "./working/idea_33/"
    submission_path = "./submission/submission.csv"
    threshold = 1.2136771218566717e-09

    # 2. Load Data
    # load_data handles alphanumeric sorting, float64 casting, and inductive preprocessing
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        cache_dir=cache_dir, load_cached_data=True
    )

    # 3. Model Training
    # ExactOASDiscriminant implements the precision_ attribute solver
    print("Initializing and fitting ExactOASDiscriminant...")
    model = ExactOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)

    # Compute Metric
    # Ensure labels match the model's classes
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)

    # REQUIRED: Print exact validation metric
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    run_failure_analysis(X_val, y_val, val_probs, model.classes_)

    # 6. Submission Logic
    if val_loss < threshold:
        print(
            f"Validation metric {val_loss} meets threshold {threshold}. Generating submission..."
        )
        test_probs = model.predict_proba(X_test)
        save_submission(test_ids, model.classes_, test_probs, submission_path)
    else:
        print(
            f"Validation metric {val_loss} does NOT meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
