import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import SEED, PROB_CLIP_EPS
from library.utils import set_seed, validate_paths
from library.pipeline import get_preprocessed_data
from library.model import train_and_evaluate, generate_submission


def main():
    # 1. Setup and Initialization
    set_seed(SEED)
    validate_paths()

    # 2. Data Loading & Preprocessing
    # Loads data, extracts geometric features, and applies the sanitized pipeline.
    # Uses caching to speed up execution if run multiple times.
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = get_preprocessed_data(
        load_cached_data=True
    )

    # 3. Model Training
    # Fits the OAS Linear Discriminant model on the training data.
    print(f"Training OASLinearDiscriminant on {len(X_train)} samples...")
    model = train_and_evaluate(X_train, y_train, X_val, y_val)

    # 4. Validation & Metric Calculation
    # We explicitly calculate the metric here to ensure compliance with the threshold check.
    print("Calculating final validation metrics...")

    # Predict probabilities on the validation set
    val_probs = model.predict_proba(X_val)

    # Apply the strict clipping required by the metric definition
    # max(min(p, 1-10^-15), 10^-15)
    val_probs_clipped = np.clip(val_probs, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)

    # Calculate Multi-class Log Loss
    # We use the model's classes to ensure correct mapping
    final_metric = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    # Print the metric in the exact required format
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis...")

    # Calculate error magnitude per sample (Cross Entropy)
    # Map string labels to integer indices
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract the predicted probability for the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Correlate error magnitude with input features to find sources of error
    # Handle potential constant features in validation subset to avoid division by zero
    correlations = []
    n_features = X_val.shape[1]

    # Calculate statistics for correlation
    y_centered = sample_losses - np.mean(sample_losses)
    y_std = np.std(sample_losses)

    if y_std > 1e-12:
        X_mean = np.mean(X_val, axis=0)
        X_centered = X_val - X_mean
        X_std = np.std(X_val, axis=0)

        # Compute covariance vector
        covariance = np.dot(y_centered, X_centered) / len(sample_losses)

        # Compute correlation vector, handling zero-variance features
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_vec = covariance / (y_std * X_std)
            corr_vec = np.nan_to_num(corr_vec)

        for i in range(n_features):
            correlations.append((i, corr_vec[i]))
    else:
        print("Validation error has zero variance. Skipping correlation analysis.")
        correlations = [(i, 0.0) for i in range(n_features)]

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature {idx}: Correlation = {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check as per requirements
    threshold = 3.058881515561734e-14

    if final_metric < threshold:
        print(f"\nValidation metric ({final_metric}) meets threshold ({threshold}).")
        print("Generating submission file...")
        generate_submission(model, X_test, test_ids)
    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
