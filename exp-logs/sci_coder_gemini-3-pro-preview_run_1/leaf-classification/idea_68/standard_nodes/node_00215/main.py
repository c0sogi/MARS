import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.preprocessing as preprocessing
import library.model as model


def run():
    # 1. Setup and Initialization
    print("Initializing Run...")
    utils.set_seed(config.SEED)

    # 2. Data Loading
    # Loads metadata and fuses tabular features with cached geometric features
    print("Loading datasets...")
    train_df, val_df, test_df = data_loader.load_datasets(load_cached_data=True)

    # Extract Test IDs for submission creation later
    test_ids = test_df[config.ID_COL].values

    # 3. Preprocessing
    # Applies VarianceThreshold -> PowerTransformer -> StandardScaler
    # Returns float64 numpy arrays
    print("Preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, classes = (
        preprocessing.get_preprocessed_data(
            train_df, val_df, test_df, load_cached_data=True
        )
    )

    print(
        f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 4. Model Training
    # Initialize the custom OAS Discriminant with centered assumption
    print("Training OASDiscriminant...")
    clf = model.OASDiscriminant(assume_centered=True)
    clf.fit(X_train, y_train)

    # 5. Validation
    print("Performing Validation...")
    # Predict probabilities (already softmaxed by the model)
    val_probs = clf.predict_proba(X_val)

    # Calculate metric using the utility function which handles clipping and rescaling
    # y_val are indices, so we pass the classes array to map them correctly if needed,
    # but utils.compute_log_loss handles the logic.
    val_metric = utils.compute_log_loss(
        y_val, val_probs, classes=np.arange(len(classes))
    )

    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Compute per-sample log loss to correlate with features
    # We manually compute the loss for the true class for each sample
    # Loss = -log(clipped_prob_of_true_class)

    # 1. Rescale and Clip (replicating utils logic for individual samples)
    row_sums = val_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    val_probs_norm = val_probs / row_sums
    val_probs_clipped = np.clip(
        val_probs_norm, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX
    )

    # 2. Extract probability of the true class
    # y_val contains indices 0..K-1
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # 3. Correlate Error with Features
    # We want to find features that correlate with high loss
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Calculate correlation between feature values and loss
        # Handle potential constant features (though they should be removed by preprocessing)
        if np.std(X_val[:, i]) > 0:
            corr = np.corrcoef(X_val[:, i], sample_losses)[0, 1]
            if not np.isnan(corr):
                correlations.append((i, corr))

    # Sort by absolute correlation (magnitude)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Validation Error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")

    # 7. Submission Generation
    # Strict threshold check as per task requirements
    THRESHOLD = 3.058881515561734e-14

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets the threshold (< {THRESHOLD})."
        )
    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet the threshold (< {THRESHOLD})."
        )
        print("Generating submission anyway to ensure output is produced.")

    print("Generating submission...")

    test_probs = clf.predict_proba(X_test)

    # Create submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, config.ID_COL, test_ids)

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run()
