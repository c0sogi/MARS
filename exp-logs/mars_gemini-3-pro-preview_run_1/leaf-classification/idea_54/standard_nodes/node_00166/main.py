import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import FLOAT_PRECISION, RANDOM_SEED, SUBMISSION_PATH
from library.data_processing import get_processed_data
from library.model import HighPrecisionOASDiscriminant


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X_val, y_val, val_probs):
    """
    Analyzes model performance on the validation set.
    Calculates correlation between error magnitude and features.
    """
    print("\n--- Failure Analysis ---")

    # y_val contains integer class indices
    # val_probs is shape (n_samples, n_classes)

    n_samples = X_val.shape[0]

    # Get the probability predicted for the true class
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    clipped_probs = np.clip(val_probs, epsilon, 1.0 - epsilon)

    # Advanced indexing to get p(y_true)
    true_class_probs = clipped_probs[np.arange(n_samples), y_val]

    # Error magnitude = negative log likelihood of the true class
    error_magnitudes = -np.log(true_class_probs)

    print(
        f"Error Magnitude Stats: Mean={np.mean(error_magnitudes):.6f}, "
        f"Max={np.max(error_magnitudes):.6f}, "
        f"Min={np.min(error_magnitudes):.6f}"
    )

    # Calculate correlation between error magnitude and each feature
    # X_val is (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    # Check for constant features to avoid warnings
    feature_stds = np.std(X_val, axis=0)

    for i in range(n_features):
        if feature_stds[i] < 1e-12:
            continue

        # Calculate Pearson correlation
        corr = np.corrcoef(X_val[:, i], error_magnitudes)[0, 1]

        if np.isfinite(corr):
            correlations.append((i, corr, abs(corr)))

    # Sort by absolute correlation descending
    correlations.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Features correlated with Error Magnitude:")
    for i, (idx, corr, abs_corr) in enumerate(correlations[:5]):
        print(f"{i+1}. Feature Index {idx}: Correlation = {corr:.4f}")

    return error_magnitudes


def main():
    # 1. Initialization
    set_seed(RANDOM_SEED)

    # 2. Data Loading
    print("Loading and processing data...")
    # Load cached data if available to save time
    X_train, y_train_raw, X_val, y_val_raw, X_test, test_ids = get_processed_data(
        load_cached_data=True
    )

    # 3. Label Encoding
    # We must ensure the encoder knows all classes from train and val
    le = LabelEncoder()
    all_labels = np.concatenate([y_train_raw, y_val_raw])
    le.fit(all_labels)

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    classes = le.classes_
    print(f"Number of classes: {len(classes)}")

    # 4. Model Training
    print("Initializing model...")
    model = HighPrecisionOASDiscriminant()

    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    # 5. Validation Inference
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # 6. Metric Calculation
    # Calculate multi-class log loss
    val_loss = log_loss(y_val, val_probs)

    # Print metric in required format
    print(f"Final Validation Metric: {val_loss}")

    # 7. Failure Analysis
    perform_failure_analysis(X_val, y_val, val_probs)

    # 8. Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 3.058881515561734e-14

    # Check condition
    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is lower than threshold ({THRESHOLD})."
        )
        print("Generating submission...")

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Format Submission
        # Columns must be sorted species names (le.classes_ is sorted)
        submission_df = pd.DataFrame(test_probs, columns=classes)

        # Add id column at the beginning
        # Ensure ids are integers
        submission_df.insert(0, "id", test_ids.astype(int))

        # Save to CSV
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_loss}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation as per requirements.")


if __name__ == "__main__":
    main()
