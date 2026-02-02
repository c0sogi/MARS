import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model


def run():
    # 1. Setup
    # Ensure reproducibility across all operations
    utils.set_seed(config.RANDOM_SEED)

    # 2. Data Loading & Preprocessing
    # library.data.load_data handles:
    # - Metadata loading
    # - Feature extraction (Moment-Completed Geometric Fusion)
    # - Caching (hashed by configuration)
    # - Inductive Preprocessing (Yeo-Johnson + Standard Scaling, fitted on Train)
    # - Returning float64 arrays for high-precision inference
    print("Loading and preprocessing data...")
    data_artifacts = data.load_data(load_cached_data=True)

    X_train = data_artifacts["X_train"]
    y_train = data_artifacts["y_train"]
    X_val = data_artifacts["X_val"]
    y_val = data_artifacts["y_val"]
    X_test = data_artifacts["X_test"]
    test_ids = data_artifacts["test_ids"]
    le = data_artifacts["label_encoder"]
    feature_names = data_artifacts["feature_names"]

    # 3. Model Training
    # Initialize the High-Precision OAS Discriminant
    # assume_centered=True is used because the model logic manually centers data
    # based on class means before covariance estimation.
    print("Training OAS Discriminant...")
    clf = model.OASDiscriminant(assume_centered=True)
    clf.fit(X_train, y_train)

    # 4. Validation
    print("Evaluating on validation set...")
    # Predict probabilities
    y_val_pred_proba = clf.predict_proba(X_val)

    # Metric Compliance:
    # "predicted probabilities are replaced with max(min(p,1-10^{-15}),10^{-15})"
    eps = 1e-15
    y_val_pred_proba_clipped = np.clip(y_val_pred_proba, eps, 1 - eps)

    # "submitted probabilities... are rescaled prior to being scored (each row is divided by the row sum)"
    # We perform this normalization to ensure the log_loss calculation matches the leaderboard metric exactly.
    row_sums = y_val_pred_proba_clipped.sum(axis=1, keepdims=True)
    y_val_pred_proba_normalized = y_val_pred_proba_clipped / row_sums

    # Compute Multi-class Log Loss
    val_metric = log_loss(y_val, y_val_pred_proba_normalized)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude per sample (Negative Log Likelihood of the true class)
    # y_val contains integer class indices. We extract the predicted probability for the true class.
    true_class_probs = y_val_pred_proba_normalized[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # Compute correlation between error magnitude and input features
    correlations = []
    # Avoid division by zero for constant features
    X_val_std = np.std(X_val, axis=0)

    for i in range(X_val.shape[1]):
        if X_val_std[i] < 1e-12:
            corr = 0.0
        else:
            # Pearson correlation
            corr = np.corrcoef(sample_losses, X_val[:, i])[0, 1]
            # Handle potential NaNs from corrcoef
            if np.isnan(corr):
                corr = 0.0
        correlations.append((feature_names[i], corr))

    # Sort features by the absolute value of their correlation with error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 3.3382359570696616e-14

    if val_metric < THRESHOLD:
        print(
            f"Validation metric {val_metric} meets threshold (< {THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        y_test_pred_proba = clf.predict_proba(X_test)

        # Apply the same Clipping and Normalization
        y_test_pred_proba_clipped = np.clip(y_test_pred_proba, eps, 1 - eps)
        row_sums_test = y_test_pred_proba_clipped.sum(axis=1, keepdims=True)
        y_test_pred_proba_normalized = y_test_pred_proba_clipped / row_sums_test

        # Construct Submission DataFrame
        # The LabelEncoder classes are sorted alphabetically, which typically matches
        # the column order required by the submission format.
        species_cols = list(le.classes_)

        submission_df = pd.DataFrame(y_test_pred_proba_normalized, columns=species_cols)

        # Insert 'id' column at the beginning
        submission_df.insert(0, "id", test_ids)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        # Save submission
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_metric} is NOT lower than threshold {THRESHOLD}."
        )
        print("Skipping submission generation as per requirements.")


if __name__ == "__main__":
    run()
