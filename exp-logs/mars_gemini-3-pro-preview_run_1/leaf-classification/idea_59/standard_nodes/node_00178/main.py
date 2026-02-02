import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config, utils, data_handler, preprocessor, model


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    print("Orchestrating Sanitized Robust-Integral High-Precision OAS Discriminant...")

    # 2. Data Loading
    # Load datasets using the handler which manages caching and geometric feature extraction
    print("Loading datasets...")
    X_train_raw, y_train, train_ids = data_handler.load_dataset(
        "train", load_cached_data=True
    )
    X_val_raw, y_val, val_ids = data_handler.load_dataset("val", load_cached_data=True)
    X_test_raw, _, test_ids = data_handler.load_dataset("test", load_cached_data=True)

    # 3. Preprocessing
    # Apply SanitizedTransformer (VarianceThreshold -> PowerTransformer -> StandardScaler)
    print("Preprocessing data...")
    X_train, X_val, X_test = preprocessor.get_transformed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    print("Training OASLinearModel...")
    clf = model.OASLinearModel()
    clf.fit(X_train, y_train)
    print(f"Model fitted. Shrinkage: {clf.shrinkage_}")

    # 5. Validation
    print("Performing validation...")
    val_probs = clf.predict_proba(X_val)

    # Compute Log Loss
    # Ensure labels are provided to handle cases where batch might miss a class (unlikely with stratified split)
    score = log_loss(y_val, val_probs, labels=clf.classes_)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {score}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Map string labels to integer indices used by the model
    class_map = {c: i for i, c in enumerate(clf.classes_)}
    y_val_idx = np.array([class_map[y] for y in y_val])

    # Calculate per-sample loss: -log(p_true)
    # Extract probability assigned to the true class
    # val_probs is (N_samples, N_classes)
    # We use advanced indexing to get p[i, y_i]
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_idx]

    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs_clipped = np.clip(true_class_probs, epsilon, 1.0)
    sample_losses = -np.log(true_class_probs_clipped)

    # Calculate correlation between features and error
    # X_val is (N, D), sample_losses is (N,)
    # We want correlation of each feature column with the loss vector

    # Center the data for fast correlation calculation
    X_centered = X_val - X_val.mean(axis=0)
    L_centered = sample_losses - sample_losses.mean()

    # Compute Covariance: (X_c.T @ L_c) / (N-1)
    # Note: X_val is numpy array from preprocessor
    N = len(sample_losses)
    cov = np.dot(X_centered.T, L_centered) / (N - 1)

    # Compute Std Devs
    X_std = X_val.std(axis=0)
    L_std = sample_losses.std()

    # Avoid division by zero
    valid_feat_mask = X_std > 0
    corrs = np.zeros(X_val.shape[1])

    if L_std > 0:
        corrs[valid_feat_mask] = cov[valid_feat_mask] / (X_std[valid_feat_mask] * L_std)

    # Identify top correlations (magnitude)
    top_indices = np.argsort(np.abs(corrs))[::-1][:5]

    print("Top 5 Features correlated with Prediction Error:")
    for idx in top_indices:
        print(f"  Feature Index {idx}: Correlation = {corrs[idx]:.4f}")

    print("------------------------\n")

    # 7. Submission
    # Threshold defined in prompt: 3.058881515561734e-14
    # Note: This threshold is extremely low (near zero).
    # We will generate the submission regardless to ensure the task "You must submit a csv file" is fulfilled.
    # We interpret the threshold requirement as a target for a perfect solution.

    target_threshold = 3.058881515561734e-14

    if score < target_threshold:
        print(f"Metric {score} meets the strict threshold {target_threshold}.")
    else:
        print(
            f"Metric {score} does not meet strict threshold {target_threshold}. Proceeding with submission for baseline evaluation."
        )

    print("Generating Test Predictions...")
    test_probs = clf.predict_proba(X_test)

    # Clip probabilities as per task spec
    test_probs = np.clip(test_probs, epsilon, 1 - epsilon)

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=clf.classes_)
    submission_df.insert(0, "id", test_ids)

    # Ensure directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save
    print(f"Saving submission to {config.OUTPUT_SUBMISSION_PATH}...")
    submission_df.to_csv(config.OUTPUT_SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    main()
