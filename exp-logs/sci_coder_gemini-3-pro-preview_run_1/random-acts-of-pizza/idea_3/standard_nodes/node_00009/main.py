import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import provided library functionality
from library.config import (
    RANDOM_SEED,
    SUBMISSION_DIR,
    SUBMISSION_FILE,
)
from library.data_processing import load_data
from library.feature_streams import generate_streams
from library.ensemble_model import HybridEnsemble


def set_seed(seed):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup and Reproducibility
    set_seed(RANDOM_SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Starting pipeline execution...")

    # 2. Data Loading
    # Loads metadata and performs feature engineering (e.g., ratio features, imputation)
    # Uses caching if available to speed up re-runs
    print("Loading and processing data...")
    train_df, val_df, test_df, feature_cols = load_data(load_cached_data=True)

    # 3. Stream Generation
    # Generates two feature streams:
    #   - Sparse Stream: Bag-of-Words + Raw Numerical (for Random Forest)
    #   - Dense Stream: Sentence Embeddings + Scaled Numerical (for Logistic Regression)
    print("Generating feature streams (Sparse & Dense)...")
    sparse_data, dense_data = generate_streams(
        train_df, val_df, test_df, feature_cols, load_cached_data=True
    )

    # Prepare target arrays
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # 4. Model Training
    # Initializes the Hybrid Ensemble and fits both internal models
    print("Initializing and training Hybrid Ensemble...")
    model = HybridEnsemble(random_state=RANDOM_SEED)

    # Fit on training data
    # We pass validation data here for internal logging, but we will compute the final metric explicitly below
    model.fit(
        X_sparse_train=sparse_data["train"],
        X_dense_train=dense_data["train"],
        y_train=y_train,
        X_sparse_val=sparse_data["val"],
        X_dense_val=dense_data["val"],
        y_val=y_val,
    )

    # 5. Validation Inference & Metric Calculation
    print("Performing validation inference...")
    # Predict probabilities on the validation set
    val_probs = model.predict_proba(sparse_data["val"], dense_data["val"])

    # Calculate ROC AUC
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate absolute error magnitude |y_true - y_prob|
    errors = np.abs(y_val - val_probs)

    # Correlate prediction errors with input numerical features to identify weaknesses
    correlations = {}
    for col in feature_cols:
        if col in val_df.columns:
            # Extract feature values, handling potential NaNs safely (though preprocessing should have handled them)
            feat_values = val_df[col].fillna(0).values

            # Compute correlation if variance exists
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    # Identify top features associated with high error
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 5 features correlated with prediction error (Failure Analysis):")
    for name, corr in sorted_corrs[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 0.648621586265928

    if val_auc > THRESHOLD:
        print(
            f"Validation metric ({val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for the test set
        test_probs = model.predict_proba(sparse_data["test"], dense_data["test"])

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": test_probs,
            }
        )

        # Save to file
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric ({val_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
