import os
import numpy as np
import pandas as pd
import sys

# Import from provided library files
from library.config import SUBMISSION_FILE, ID_COL
from library.utils import set_seed, compute_log_loss
from library.data import load_and_merge_data
from library.pipeline import run_pipeline
from library.model import OASDiscriminant


def main():
    # 1. Setup
    set_seed(42)
    print("=== Orchestrating Sanitized Orthogonal-Geometric OAS Discriminant ===")

    # 2. Data Loading
    # Load raw data (tabular + geometric features extracted on the fly/cached)
    print("Loading data...")
    X_train_raw, y_train, _ = load_and_merge_data("train", load_cached_data=True)
    X_val_raw, y_val, _ = load_and_merge_data("val", load_cached_data=True)
    X_test_raw, _, test_ids = load_and_merge_data("test", load_cached_data=True)

    # 3. Pipeline Processing
    # Transform features (VarianceThreshold -> Yeo-Johnson -> StandardScaler)
    print("Running pipeline...")
    X_train, X_val, X_test = run_pipeline(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    print("Training OAS Discriminant...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # 5. Validation
    print("Validating...")
    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)

    # Convert string labels to integer indices for metric calculation
    y_val_indices = model.le_.transform(y_val)

    # Compute Metric
    val_loss = compute_log_loss(y_val_indices, val_probs)
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude per sample (Negative Log Likelihood of the true class)
    # Avoid log(0) by clipping, though model output is softmax so usually > 0
    # prob_true[i] = val_probs[i, y_true[i]]
    N = len(y_val)
    prob_true = val_probs[np.arange(N), y_val_indices]
    prob_true = np.clip(prob_true, 1e-15, 1.0)
    error_magnitudes = -np.log(prob_true)

    # Compute correlation between error magnitude and each feature in X_val
    # X_val is (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vec = X_val[:, i]
        # Handle constant features if any slipped through (unlikely due to pipeline)
        if np.std(feature_vec) < 1e-12:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vec, error_magnitudes)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most positively correlated with error
    # (High feature value -> High Error)
    # We also look at absolute correlation to find any strong relationship
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features correlated with Error Magnitude:")
    for idx in top_indices:
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.6f}")

    # 7. Submission
    # Threshold defined in task
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        print(
            f"Validation metric {val_loss} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Construct Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, ID_COL, test_ids)

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric {val_loss} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
