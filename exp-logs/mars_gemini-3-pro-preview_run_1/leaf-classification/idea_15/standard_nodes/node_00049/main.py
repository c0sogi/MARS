import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library.config import (
    SEED,
    CLIP_EPSILON,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
    FEATURE_COLS,
)
from library.preprocessing import get_preprocessed_data
from library.oas_discriminant import OASLinearDiscriminant


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_per_sample_log_loss(y_true, y_prob, labels):
    """
    Calculates log loss for each sample individually.
    y_true: array of shape (n_samples,)
    y_prob: array of shape (n_samples, n_classes)
    labels: list of class labels corresponding to y_prob columns
    """
    # Create a mapping from label to column index
    label_to_idx = {label: i for i, label in enumerate(labels)}

    losses = []
    for i, true_label in enumerate(y_true):
        if true_label in label_to_idx:
            idx = label_to_idx[true_label]
            prob = y_prob[i, idx]
            # Clip probability
            prob = max(min(prob, 1 - CLIP_EPSILON), CLIP_EPSILON)
            loss = -np.log(prob)
        else:
            # Should not happen in this dataset context
            loss = -np.log(CLIP_EPSILON)
        losses.append(loss)
    return np.array(losses)


def run():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # Using cached preprocessed data to ensure float64 precision and speed
    print("Loading preprocessed data...")
    (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test) = (
        get_preprocessed_data(load_cached_data=True)
    )

    # 3. Validation Training
    print("Training model on training set...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # Inference on Validation
    # Note: The provided model is CPU-based (numpy/sklearn).
    # GPU utilization logic is skipped as it is not applicable to this specific
    # scikit-learn based implementation provided in the library.
    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)

    # Clip probabilities
    val_probs_clipped = np.clip(val_probs, CLIP_EPSILON, 1 - CLIP_EPSILON)

    # Calculate Metric
    # Ensure labels provided to log_loss match the model's classes
    metric = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate per-sample loss
    sample_losses = calculate_per_sample_log_loss(
        y_val, val_probs_clipped, model.classes_
    )

    # Correlate loss with features
    correlations = []
    # X_val is a numpy array, we assume column order matches FEATURE_COLS
    for i, feature_name in enumerate(FEATURE_COLS):
        # Check for constant features to avoid warnings
        if np.std(X_val[:, i]) > 0:
            corr, _ = pearsonr(X_val[:, i], sample_losses)
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0
        correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error (Log Loss):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission Logic
    # Threshold defined in task
    THRESHOLD = 1.2136771218566717e-09

    if metric < THRESHOLD:
        print(
            f"\nMetric {metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Combine Train and Val
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        print(f"Retraining on full dataset ({X_full.shape[0]} samples)...")
        final_model = OASLinearDiscriminant()
        final_model.fit(X_full, y_full)

        print("Predicting on test set...")
        test_probs = final_model.predict_proba(X_test)
        test_probs_clipped = np.clip(test_probs, CLIP_EPSILON, 1 - CLIP_EPSILON)

        # Create Submission DataFrame
        # Columns: id, then species (sorted alphanumerically)
        # model.classes_ comes from np.unique(y), which sorts alphanumerically
        submission_columns = [ID_COL] + list(final_model.classes_)

        # Prepare data
        # ids_test needs to be reshaped for concatenation or inserted into DF
        df_sub = pd.DataFrame(test_probs_clipped, columns=final_model.classes_)
        df_sub.insert(0, ID_COL, ids_test)

        # Save
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {metric} did NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
