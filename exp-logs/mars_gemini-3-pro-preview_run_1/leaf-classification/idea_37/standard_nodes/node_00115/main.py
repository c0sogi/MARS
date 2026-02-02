import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library import config, data_loader, preprocessing, model


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_failure_analysis(X_val, y_val, val_probs, feature_names=None):
    """
    Performs failure analysis by correlating error magnitude with features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate per-sample log loss (cross-entropy)
    # y_val is label encoded integers.
    # We need the probability assigned to the true class.

    # Create an array of probabilities for the true classes
    # val_probs shape: (n_samples, n_classes)
    # y_val shape: (n_samples,)

    # Advanced indexing to get p(y_true)
    true_class_probs = val_probs[np.arange(len(y_val)), y_val]

    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)

    # Error magnitude is the negative log likelihood
    errors = -np.log(true_class_probs)

    print(f"Mean Error (Log Loss): {np.mean(errors)}")
    print(f"Max Error: {np.max(errors)}")

    # Correlate errors with features
    # X_val is a numpy array (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Skip constant columns if any (though pipeline removes them)
        if np.std(feature_col) == 0:
            continue

        corr, _ = pearsonr(errors, feature_col)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features correlated with Error:")
    for idx, corr in correlations[:5]:
        feat_name = f"Feature_{idx}"
        if feature_names is not None and idx < len(feature_names):
            feat_name = feature_names[idx]
        print(f"{feat_name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(42)
    print("Starting execution...")

    # 2. Data Loading
    # This handles loading metadata, extracting image features, and fusing them.
    # It returns DataFrames/Arrays.
    print("Loading and augmenting data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, class_names = (
        data_loader.load_and_augment_data(load_cached_data=True)
    )

    # Keep track of feature names for analysis before converting to numpy
    feature_names = list(X_train_raw.columns)

    # 3. Preprocessing
    # Applies Yeo-Johnson and Scaling in float64
    print("Preprocessing data...")
    X_train, X_val, X_test = preprocessing.process_and_cache_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    print("Training Augmented OAS Discriminant...")
    clf = model.OASLinearDiscriminant(assume_centered=config.OAS_ASSUME_CENTERED)
    clf.fit(X_train, y_train)

    # 5. Validation
    print("Predicting on validation set...")
    val_probs = clf.predict_proba(X_val)

    # Compute Metric
    # Note: sklearn log_loss handles the log(0) case internally with an epsilon,
    # but the task description specifies a specific clipping for submission.
    # For the validation metric calculation here, we use the standard implementation
    # to assess model performance accurately.
    val_loss = log_loss(y_val, val_probs, labels=clf.classes_)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    run_failure_analysis(X_val, y_val, val_probs, feature_names)

    # 7. Submission Logic
    # Threshold defined in task
    THRESHOLD = 1.7583657710772332e-11

    if val_loss < THRESHOLD:
        print(
            f"\nValidation score ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test
        test_probs = clf.predict_proba(X_test)

        # Apply the specific clipping mentioned in the task description for submission
        # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
        test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

        # Construct Submission DataFrame
        # The model.classes_ are indices 0..98 which map to class_names
        submission_df = pd.DataFrame(test_probs, columns=class_names)
        submission_df.insert(0, "id", test_ids)

        # Save
        save_path = config.SUBMISSION_FILE_PATH
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation score ({val_loss}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
