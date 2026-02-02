import sys
import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_dataset
from library.preprocessor import preprocess_data
from library.fisher_gp_model import train_fisher_gp, predict_and_submit


def run_failure_analysis(model, X_val, y_val, feature_names=None):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("\n=== Failure Analysis ===")

    # 1. Compute Error Magnitude
    # We define error magnitude as (1 - probability_of_true_class)
    probs = model.predict_proba(X_val)

    # Extract probability assigned to the true class
    # y_val contains integer class indices
    true_class_probs = probs[np.arange(len(y_val)), y_val]

    # Error magnitude: 0 means perfect prediction, 1 means total failure
    error_magnitude = 1.0 - true_class_probs

    # 2. Calculate Correlations
    # We calculate Pearson correlation between each feature column and the error vector
    correlations = []
    n_features = X_val.shape[1]

    # Handle NaNs in correlation calculation (e.g., constant features)
    for i in range(n_features):
        feature_col = X_val[:, i]
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, error_magnitude)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        correlations.append(corr)

    correlations = np.array(correlations)

    # 3. Report Top Correlated Features
    # Sort by absolute correlation
    top_indices = np.argsort(np.abs(correlations))[::-1][:10]

    print("Top 10 features correlated with error magnitude:")
    for idx in top_indices:
        feat_name = f"Feature_{idx}"
        # If we had feature names passed, we could use them.
        # Based on data_loader, features are margin_1..64, shape_1..64, texture_1..64
        # We can reconstruct the name roughly.
        if idx < 64:
            feat_name = f"margin_{idx+1}"
        elif idx < 128:
            feat_name = f"shape_{idx-64+1}"
        else:
            feat_name = f"texture_{idx-128+1}"

        print(f"  {feat_name}: {correlations[idx]:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Load Data
    # load_cached_data=True allows using previously processed .npy files if available
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=True
    )

    # 3. Preprocess Data
    # Applies PowerTransformer (Yeo-Johnson) and StandardScaler
    print("Preprocessing data...")
    X_train_proc, X_val_proc, X_test_proc = preprocess_data(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Train and Validate
    # train_fisher_gp handles fitting the LDA backbone and GPC head,
    # and computing the log loss on the validation set.
    print("Training model...")
    model, val_loss = train_fisher_gp(
        X_train_proc, y_train, X_val_proc, y_val, verbose=True
    )

    # 5. Print Final Metric
    # Requirement: Print full precision without rounding
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    run_failure_analysis(model, X_val_proc, y_val)

    # 7. Submission
    # Threshold defined in task description
    THRESHOLD = 1.4705447816556679e-08

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, X_test_proc, test_ids, classes)
    else:
        print(
            f"\nValidation metric ({val_loss}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
