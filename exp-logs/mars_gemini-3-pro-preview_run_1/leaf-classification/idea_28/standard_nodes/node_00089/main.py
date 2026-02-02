import os
import sys
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr

# Import provided library modules
from library import config, preprocessing, factorized_lda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_log_loss(y_true_indices, probs):
    """
    Calculates the multi-class log loss.
    probs: shape (n_samples, n_classes)
    y_true_indices: shape (n_samples,) containing integer class indices
    """
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)

    # Normalize rows to sum to 1 (as per task description, though softmax usually does this)
    probs_clipped /= probs_clipped.sum(axis=1, keepdims=True)

    # Select probabilities corresponding to true classes
    n_samples = len(y_true_indices)
    true_class_probs = probs_clipped[np.arange(n_samples), y_true_indices]

    # Compute log loss
    log_loss = -np.mean(np.log(true_class_probs))
    return log_loss


def perform_failure_analysis(X_val, y_val_indices, probs_val):
    """
    Correlates error magnitude with input features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude: 1 - probability of the true class
    n_samples = len(y_val_indices)
    true_class_probs = probs_val[np.arange(n_samples), y_val_indices]
    error_magnitude = 1.0 - true_class_probs

    # X_val is already a flat matrix
    # Compute correlations
    correlations = []
    for i in range(X_val.shape[1]):
        feature_vec = X_val[:, i]
        # Handle constant features to avoid division by zero in correlation
        if np.std(feature_vec) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_vec, error_magnitude)
        correlations.append((f"feature_{i}", corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with prediction error:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.4f}")

    return correlations


def main():
    # 1. Setup
    set_seed(config.RANDOM_SEED)

    # 2. Data Loading
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = (
        preprocessing.get_preprocessed_data(
            load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE
        )
    )

    # 3. Model Training
    print("Training Global OAS LDA model...")
    model = factorized_lda.GlobalOASLDA()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation...")
    probs_val = model.predict_proba(X_val)

    # Convert string labels to indices for metric calculation
    y_val_indices = model.le.transform(y_val)

    # Calculate Metric
    val_metric = calculate_log_loss(y_val_indices, probs_val)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(X_val, y_val_indices, probs_val)

    # 6. Submission
    # Threshold check as per requirements
    threshold = 1.2136771218566717e-09

    if val_metric < threshold:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({threshold}). Generating submission..."
        )

        # Generate predictions
        probs_test = model.predict_proba(X_test)

        # Create Submission DataFrame
        submission = pd.DataFrame(probs_test, columns=model.classes_)
        submission.insert(0, config.ID_COLUMN, ids_test)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
