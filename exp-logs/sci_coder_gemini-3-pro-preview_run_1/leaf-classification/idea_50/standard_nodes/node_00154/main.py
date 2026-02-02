import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import spearmanr

# Import from provided libraries
from library.config import SEED, ID_COL, SUBMISSION_DIR
from library.data_loader import load_data
from library.model import ParsimoniousOASClassifier


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def perform_failure_analysis(X_val, y_val, val_probs, classes):
    """
    Analyzes the correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")

    # 1. Calculate per-sample Log Loss (Error Magnitude)
    # Map class labels to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract predicted probability for the true class
    # Clip to avoid log(0)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_indices]

    # Error magnitude = Negative Log Likelihood
    error_magnitudes = -np.log(true_class_probs)

    print(f"Mean Error Magnitude: {np.mean(error_magnitudes):.6e}")
    print(f"Max Error Magnitude:  {np.max(error_magnitudes):.6e}")

    # 2. Correlate Error with Features
    correlations = []
    feature_names = X_val.columns.tolist()

    # Calculate Spearman correlation for each feature
    # (Spearman is robust to outliers and non-linear monotonic relationships)
    for feature in feature_names:
        feat_values = X_val[feature].values
        corr, _ = spearmanr(feat_values, error_magnitudes)
        # Handle NaN correlations (e.g., constant features)
        if np.isnan(corr):
            corr = 0.0
        correlations.append((feature, abs(corr), corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 Features correlated with Error Magnitude:")
    for name, abs_corr, raw_corr in correlations[:5]:
        print(f"  {name}: {raw_corr:.4f} (Abs: {abs_corr:.4f})")


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # load_data handles:
    # - Feature Engineering (Geometric Scalars)
    # - Subtractive Fusion (Dropping Shape, Adding Geometric)
    # - Inductive Preprocessing (Yeo-Johnson + StandardScaler)
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(load_cached_data=True)

    # 3. Model Training
    # ParsimoniousOASClassifier uses analytical shrinkage in float64
    print("\nInitializing and fitting ParsimoniousOASClassifier...")
    model = ParsimoniousOASClassifier(assume_centered=True)
    model.fit(X_train, y_train)

    # 4. Validation
    print("Predicting on Validation set...")
    val_probs = model.predict_proba(X_val)

    # Clip probabilities as per task description for metric calculation
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)

    # Calculate Multi-class Log Loss
    # Ensure labels match the order of columns in val_probs (model.classes_)
    metric = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    # Print full precision metric as required
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    perform_failure_analysis(X_val, y_val, val_probs, model.classes_)

    # 6. Submission Generation
    # Strict threshold check
    THRESHOLD = 3.3382359570696616e-14

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test set
        test_probs = model.predict_proba(X_test)
        test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)

        # Format Submission
        submission_df = pd.DataFrame(test_probs_clipped, columns=model.classes_)
        submission_df.insert(0, ID_COL, test_ids.values)

        # Save
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
