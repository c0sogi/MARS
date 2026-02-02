import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import warnings

# Import from the provided library files
from library.config import OUTPUT_DIR, RANDOM_SEED, FEATURES
from library.data_manager import load_dataset
from library.robust_classifier import GeometricOASDiscriminant


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    warnings.filterwarnings("ignore")

    print("Initializing orchestration script...")

    # 2. Load Data
    # Utilizing the cached, preprocessed data (float64, Yeo-Johnson + StandardScaled)
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=True
    )

    # 3. Model Training
    # The GeometricOASDiscriminant uses Geometric Median for centroids and OAS for covariance
    print("Training GeometricOASDiscriminant...")
    model = GeometricOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # labels parameter ensures correct mapping even if y_val doesn't contain all classes
    val_metric = log_loss(y_val, val_probs, labels=range(len(classes)))

    # Print the required metric line with full precision
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample loss (error magnitude)
    # We extract the probability assigned to the true class
    # y_val are integer indices, val_probs is (n_samples, n_classes)
    row_indices = np.arange(len(y_val))
    true_class_probs = val_probs[row_indices, y_val]
    # Clip for numerical stability in log calculation (though log_loss handles this internally for the metric)
    true_class_probs = np.maximum(true_class_probs, 1e-15)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between features and error magnitude
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Handle potential constant features (std=0) to avoid NaN correlation
        if np.std(X_val[:, i]) > 0 and np.std(sample_losses) > 0:
            corr = np.corrcoef(X_val[:, i], sample_losses)[0, 1]
            correlations.append((FEATURES[i], corr))
        else:
            correlations.append((FEATURES[i], 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude (systematic error patterns):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission
    # Strict threshold check as per requirements
    THRESHOLD = 1.2136771218566717e-09

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate Test Predictions
        test_probs = model.predict_proba(X_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save to file
        submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({val_metric}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
