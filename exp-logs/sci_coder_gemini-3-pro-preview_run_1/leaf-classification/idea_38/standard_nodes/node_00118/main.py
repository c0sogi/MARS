import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from the provided library modules
from library.config import (
    SEED,
    SUBMISSION_PATH,
    TRAIN_CSV,
    SORT_COLUMNS,
    GEOMETRIC_FEATURES,
)
from library.preprocessing import get_preprocessed_data
from library.model import OASLinearDiscriminant

# Set global random seed
np.random.seed(SEED)


def main():
    # 1. Load Data
    # We use load_cached_data=True to utilize the pre-computed features and pipeline
    # stored in the working directory, ensuring fast execution.
    print("Loading preprocessed data...")
    try:
        (train_data, val_data, test_data) = get_preprocessed_data(load_cached_data=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, _, ids_test = test_data

    # 2. Model Training
    print(f"Training with feature set size: {X_train.shape[1]}")
    print("Initializing and training OASLinearDiscriminant model...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # 3. Validation
    print("Performing validation...")
    # Filter validation set to ensure we only evaluate on classes known to the model
    # (Though in this dataset split, train and val usually share all classes)
    valid_mask = np.isin(y_val, model.classes_)
    X_val_filtered = X_val[valid_mask]
    y_val_filtered = y_val[valid_mask]

    if len(y_val_filtered) == 0:
        print("Error: Validation set has no matching classes with training set.")
        return

    # Predict probabilities
    val_probs = model.predict_proba(X_val_filtered)

    # Calculate Log Loss
    val_loss = log_loss(y_val_filtered, val_probs, labels=model.classes_)

    # Print the required metric string
    print(f"Final Validation Metric: {val_loss}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude per sample (Negative Log Likelihood of the true class)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val_filtered])

    # Extract probability of the true class
    # val_probs is shape (n_samples, n_classes)
    prob_true = val_probs[np.arange(len(y_val_filtered)), y_val_indices]

    # Clip to avoid log(0) - though predict_proba already clips, we ensure safety here
    prob_true = np.clip(prob_true, 1e-15, 1.0)
    error_magnitude = -np.log(prob_true)

    # Reconstruct Feature Names
    # We need to match the order used in data_loader.py
    # 1. Load metadata columns (excluding non-features)
    df_meta_header = pd.read_csv(TRAIN_CSV, nrows=0)
    non_feature_cols = {"id", "species", "file_path", "full_path"}
    meta_features = [c for c in df_meta_header.columns if c not in non_feature_cols]

    # 2. Combine with geometric features
    all_features = meta_features + GEOMETRIC_FEATURES

    # 3. Apply sorting if configured
    if SORT_COLUMNS:
        all_features = sorted(all_features)

    # Verify feature count matches
    if len(all_features) != X_val_filtered.shape[1]:
        print(
            f"Warning: Feature name count ({len(all_features)}) does not match data dimensions ({X_val_filtered.shape[1]}). Using generic names."
        )
        feature_names = [f"Feature_{i}" for i in range(X_val_filtered.shape[1])]
    else:
        feature_names = all_features

    # Calculate Pearson correlation between each feature and the error magnitude
    correlations = []
    for i, feature_name in enumerate(feature_names):
        feature_values = X_val_filtered[:, i]

        # Skip constant features to avoid warnings
        if np.std(feature_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_values, error_magnitude)

        if np.isnan(corr):
            corr = 0.0

        correlations.append((feature_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 5.234670549314967e-14

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", ids_test)

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_loss}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
