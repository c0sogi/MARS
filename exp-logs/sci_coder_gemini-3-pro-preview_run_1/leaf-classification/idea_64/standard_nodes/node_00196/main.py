import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import SEED, SUBMISSION_FILE, ID_COL
from library.utils import set_seed, compute_log_loss
from library.data_loader import load_and_process_data
from library.preprocessing import preprocess_data
from library.model import OASDiscriminant


def run_pipeline():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    print("Loading and processing data...")
    # load_cached_data=True allows using the ./working cache if available
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process_data(
        load_cached_data=True
    )

    # 3. Preprocess Data
    print("Preprocessing features...")
    # Fits on X_train, transforms all. Handles caching.
    X_train_trans, X_val_trans, X_test_trans = preprocess_data(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Train Model
    print("Training OAS Discriminant on training set...")
    model = OASDiscriminant()
    model.fit(X_train_trans, y_train)

    # 5. Validation
    print("Validating...")
    val_probs = model.predict_proba(X_val_trans)

    # Compute metric
    val_loss = compute_log_loss(y_val, val_probs, model.classes_)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {val_loss:.20f}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Calculate per-sample loss
    # We need to identify the probability assigned to the true class for each sample
    class_map = {c: i for i, c in enumerate(model.classes_)}
    y_val_indices = np.array([class_map[label] for label in y_val])

    # Gather probabilities of the true classes
    # prob[i, true_class_index]
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_indices]

    # Clip probabilities to avoid log(0) - matching the metric logic
    eps = 1e-15
    true_class_probs_clipped = np.clip(true_class_probs, eps, 1 - eps)

    # Loss per sample = -log(p_true)
    sample_losses = -np.log(true_class_probs_clipped)

    # Correlate features with error
    n_features = X_val_trans.shape[1]
    correlations = []

    # Calculate Pearson correlation for each feature against the loss vector
    # Handle potential constant features in validation set which would cause division by zero in correlation
    for i in range(n_features):
        feature_vals = X_val_trans[:, i]
        if np.std(feature_vals) > 1e-9:
            corr, _ = pearsonr(feature_vals, sample_losses)
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        correlations.append((i, abs(corr), corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 features correlated with error (Index, Abs Corr, Raw Corr):")
    for idx, abs_c, raw_c in correlations[:5]:
        print(f"Feature {idx}: {raw_c:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Retrain on full data (Train + Val)
        print("Retraining on combined dataset...")
        X_full = np.concatenate([X_train_trans, X_val_trans], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        model_full = OASDiscriminant()
        model_full.fit(X_full, y_full)

        # Predict on Test
        print("Predicting on test set...")
        test_probs = model_full.predict_proba(X_test_trans)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model_full.classes_)
        submission_df.insert(0, ID_COL, test_ids)

        # Save
        print(f"Saving submission to {SUBMISSION_FILE}...")
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
