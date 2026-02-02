import os
import sys
import numpy as np
import pandas as pd
from library.config import SEED, FEATURE_COLS
from library.preprocessor import process_and_cache_data
from library.model import OASLinearDiscriminant
from library.utils import calculate_log_loss, save_submission


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_failure_analysis(X_val, y_val, val_probas):
    """
    Analyzes model failures by correlating error magnitude with input features.
    """
    print("\nStarting Failure Analysis...")

    # 1. Calculate Error Magnitude per sample (Negative Log Likelihood of the true class)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    # Get the probability assigned to the true class
    # y_val are indices 0..K-1
    true_class_probs = val_probas[np.arange(len(y_val)), y_val]
    clipped_probs = np.clip(true_class_probs, epsilon, 1.0)
    error_magnitude = -np.log(clipped_probs)

    # 2. Correlate Error Magnitude with Features
    # We compute the Pearson correlation coefficient between each feature column and the error vector
    n_features = X_val.shape[1]
    correlations = []

    # Centering for correlation calculation
    error_centered = error_magnitude - np.mean(error_magnitude)
    error_norm = np.linalg.norm(error_centered)

    if error_norm == 0:
        print("Error magnitude is constant (likely 0). Skipping correlation analysis.")
        return

    for i in range(n_features):
        feature_col = X_val[:, i]
        feature_centered = feature_col - np.mean(feature_col)
        feature_norm = np.linalg.norm(feature_centered)

        if feature_norm == 0:
            corr = 0.0
        else:
            corr = np.dot(feature_centered, error_centered) / (
                feature_norm * error_norm
            )

        correlations.append((FEATURE_COLS[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load and Preprocess Data
    # This uses the library function which handles caching and float64 transformation
    print("Loading and preprocessing data...")
    data = process_and_cache_data(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    ids_test = data["ids_test"]
    classes = data["classes"]

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Initialize and Train Model
    print("Initializing OASLinearDiscriminant model...")
    model = OASLinearDiscriminant()

    print("Training model...")
    model.fit(X_train, y_train)

    # 4. Validation
    print("Running validation inference...")
    val_probas = model.predict_proba(X_val)

    # Calculate metric
    val_metric = calculate_log_loss(y_val, val_probas)

    # Print EXACTLY as requested
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    run_failure_analysis(X_val, y_val, val_probas)

    # 6. Submission Logic
    # Threshold defined in task description
    THRESHOLD = 1.2136771218566717e-09

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        test_probas = model.predict_proba(X_test)
        save_submission(
            ids_test,
            test_probas,
            classes,
            output_dir="./submission",
            filename="submission.csv",
        )
    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
