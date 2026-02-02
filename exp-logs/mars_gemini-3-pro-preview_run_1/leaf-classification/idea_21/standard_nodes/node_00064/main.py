import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import sys

# Import from the provided library
from library.config import SEED, SUBMISSION_PATH
from library.utils import set_seed, save_submission
from library.preprocessing import get_preprocessed_data
from library.model import LinearizedOASClassifier


def calculate_competition_metric(y_true, y_pred_proba, classes):
    """
    Calculates the Multi-class Log Loss with specific clipping and normalization
    as defined in the task description.
    """
    # 1. Rescale: Divide each row by the row sum (ensure sum to 1)
    # Note: Softmax usually ensures this, but we enforce it as per instructions.
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums

    # 2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn log_loss handles the string labels if we provide the classes list
    return log_loss(y_true, y_pred_proba, labels=classes)


def perform_failure_analysis(X_val, y_val, y_pred_proba, classes):
    """
    Analyzes systematic errors by correlating per-sample loss with feature values.
    """
    print("\n--- Failure Analysis ---")

    # Map class names to column indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Calculate per-sample log loss
    # Loss_i = -log(p_{true_class})
    # We use the clipped probabilities for consistency
    epsilon = 1e-15
    y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_pred_proba_clipped = y_pred_proba_clipped / y_pred_proba_clipped.sum(
        axis=1, keepdims=True
    )

    p_true = y_pred_proba_clipped[np.arange(len(y_val)), y_val_indices]
    sample_losses = -np.log(p_true)

    print(f"Mean Validation Loss: {np.mean(sample_losses):.10f}")
    print(f"Max Validation Loss:  {np.max(sample_losses):.10f}")

    # Calculate correlation between feature values and error (loss)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Skip constant features to avoid warnings
        if np.std(feature_vals) < 1e-12:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_vals, sample_losses)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most correlated with error (magnitude)
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 features associated with prediction error:")
    for idx in top_indices:
        print(f"  Feature {idx}: Correlation = {correlations[idx]:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)
    print("Initializing workflow...")

    # 2. Data Loading
    # get_preprocessed_data handles loading, Yeo-Johnson transform, Standard Scaling, and caching.
    # It returns data in float64 precision.
    print("Loading and preprocessing datasets...")
    (train_data, val_data, test_data) = get_preprocessed_data(load_cached_data=True)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # 3. Model Training
    print("\nTraining Linearized OAS Classifier...")
    model = LinearizedOASClassifier()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Validation
    print("\nRunning validation inference...")
    val_probs = model.predict_proba(X_val)

    # Calculate Metric
    metric = calculate_competition_metric(y_val, val_probs, model.classes_)

    # REQUIRED: Print the final validation metric in the exact format
    print(f"Final Validation Metric: {metric:.20f}")

    # 5. Failure Analysis
    perform_failure_analysis(X_val, y_val, val_probs, model.classes_)

    # 6. Submission
    # Threshold defined in the task description
    THRESHOLD = 1.2136771218566717e-09

    if metric < THRESHOLD:
        print(f"\nValidation metric ({metric}) meets the threshold (< {THRESHOLD}).")
        print("Generating submission for test set...")

        test_probs = model.predict_proba(X_test)

        # Save submission
        save_submission(ids_test, test_probs, list(model.classes_), SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({metric}) does NOT meet the threshold (< {THRESHOLD})."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    main()
