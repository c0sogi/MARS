import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import from provided libraries
from library.config import SEED
from library.utils import set_seed
from library.preprocessing import get_preprocessed_data
from library.model import train_model, generate_submission


def main():
    # 1. Setup and Reproducibility
    set_seed(SEED)

    # 2. Load Data
    # Uses the 'Sanitized Pipeline': VarianceThreshold -> Yeo-Johnson -> StandardScaler
    # Data is returned as float64 numpy arrays
    data = get_preprocessed_data(load_cached_data=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]

    # 3. Model Training
    # Trains the OAS Discriminant (Linear Discriminant with Oracle Approximating Shrinkage)
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    # We explicitly calculate the metric on the validation set to ensure precision
    val_probs = model.predict_proba(X_val)
    metric = log_loss(y_val, val_probs, labels=model.classes_)

    # Print the required metric format
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Map string labels to integer indices matching the model's classes
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Calculate per-sample log loss (Error Magnitude)
    # prob of true class
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_indices]
    # Clip for safety in log calculation (though model output is already clipped)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between Error Magnitude and Input Features
    # X_val is (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vec = X_val[:, i]
        # Handle constant features if any slipped through (though VarianceThreshold should catch them)
        if np.std(feature_vec) == 0:
            corr = 0
        else:
            corr = np.corrcoef(feature_vec, sample_losses)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features associated with Error Magnitude (Index: Correlation):")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check as per instructions
    threshold = 3.058881515561734e-14

    if metric < threshold:
        generate_submission(model, X_test, test_ids)
    else:
        print(
            f"\nValidation metric {metric} did not meet the threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
