import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib

import library.utils

# Cite debug_lesson_1: Force Reload Modified Modules in Persistent Sessions
importlib.reload(library.utils)
from library.utils import set_seed, compute_metric
from library.data_loader import load_dataset
from library.model import SanitizedOASDiscriminant


def run():
    # 1. Set Seed for Reproducibility
    set_seed(42)

    # 2. Device Detection (Requirement Check)
    # Although the provided model is CPU-based (Numpy/Scikit-Learn),
    # we detect the device as per requirements.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {device}")

    # 3. Load Data
    # load_cached_data=True ensures we use the pre-processed features from ./working
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=True
    )

    # 4. Initialize and Train Model
    print("Initializing and training SanitizedOASDiscriminant...")
    model = SanitizedOASDiscriminant()
    model.fit(X_train, y_train)

    # 5. Validation Inference
    print("Performing validation inference...")
    val_probs = model.predict_proba(X_val)

    # 6. Compute Validation Metric
    # y_val contains class indices (0 to K-1).
    # We pass labels=range(len(classes)) to ensure log_loss handles the mapping correctly.
    labels = list(range(len(classes)))
    val_loss = compute_metric(y_val, val_probs, labels=labels)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    # Compute per-sample log loss (error magnitude)
    # Clip probabilities as done in the metric to avoid log(0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    # Normalize rows (though model output is softmax, metric does this too)
    row_sums = val_probs_clipped.sum(axis=1, keepdims=True)
    val_probs_norm = val_probs_clipped / row_sums

    # Extract probability assigned to the true class
    # y_val are indices
    sample_indices = np.arange(len(y_val))
    true_class_probs = val_probs_norm[sample_indices, y_val]

    # Error is negative log likelihood
    errors = -np.log(true_class_probs)

    # Compute correlation between each feature and the error
    n_features = X_val.shape[1]
    correlations = []

    # Handle potential constant features in X_val which would cause division by zero in correlation
    X_val_std = np.std(X_val, axis=0)

    for i in range(n_features):
        if X_val_std[i] < 1e-12:
            correlations.append(0.0)
        else:
            # Correlation between feature column i and error vector
            corr = np.corrcoef(X_val[:, i], errors)[0, 1]
            # Handle NaN if any numerical issue occurred
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most correlated with error (magnitude)
    # We look at absolute correlation to find features that explain error variance
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 features correlated with error magnitude:")
    for idx in top_indices:
        print(f"Feature Index {idx}: Correlation {correlations[idx]:.4f}")

    # 8. Submission Generation
    # Strict threshold check
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        print(
            f"Validation metric {val_loss} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Prepare Submission DataFrame
        df_sub = pd.DataFrame(test_probs, columns=classes)
        df_sub.insert(0, "id", test_ids)

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"Validation metric {val_loss} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation as per requirements.")


if __name__ == "__main__":
    run()
