import numpy as np
import pandas as pd
import sys
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import library functions
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets, get_combined_train_data
from library.preprocessor import preprocess_data
from library.model_definitions import train_logreg_cv, train_lda, train_gpc
from library.config import SUBMISSION_OUTPUT_PATH

# Constants
VALIDATION_THRESHOLD = 0.010054905410813797


def perform_failure_analysis(X_val, y_val, probs, classes):
    """
    Analyzes the correlation between feature values and prediction error (log loss).
    """
    print("\nPerforming Failure Analysis...")

    # Map string labels to indices
    class_map = {label: i for i, label in enumerate(classes)}
    try:
        y_indices = np.array([class_map[label] for label in y_val])
    except KeyError as e:
        print(f"Error in label mapping during failure analysis: {e}")
        return

    # Calculate per-sample log loss
    # Select the probability assigned to the true class
    true_class_probs = probs[np.arange(len(y_val)), y_indices]

    # Clip to avoid log(0) and align with metric definition
    true_class_probs = np.clip(true_class_probs, 1e-15, 1 - 1e-15)
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Sample Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Sample Loss: {np.max(sample_losses):.6f}")

    # Calculate correlation between each feature and the loss
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Handle potential constant features which produce NaN correlation
        if np.std(feature_vals) == 0:
            corr = 0
        else:
            corr = np.corrcoef(feature_vals, sample_losses)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"  Feature {idx}: Correlation = {corr:.4f}")


def main():
    # 1. Setup
    set_seed()
    print("Starting execution...")

    # 2. Load Data for Validation
    # We load the split data to perform proper validation
    (X_train, y_train, _), (X_val, y_val, _), _ = load_datasets(load_cached=True)

    # 3. Preprocessing (Validation Phase)
    print("\n--- Preprocessing (Validation Phase) ---")

    # Pipeline A: Scaling only (for Linear Models)
    # We fit on X_train and transform X_val
    X_train_scaled, X_val_scaled, _ = preprocess_data(
        X_train, X_val, None, use_pca=False, cache_prefix="val_split", load_cached=True
    )

    # Pipeline B: Scaling + PCA (for GPC)
    # We fit on X_train and transform X_val
    X_train_pca, X_val_pca, _ = preprocess_data(
        X_train, X_val, None, use_pca=True, cache_prefix="val_split", load_cached=True
    )

    # 4. Model Training (Validation Phase)
    print("\n--- Training Models (Validation Phase) ---")

    # Train Logistic Regression
    model_lr = train_logreg_cv(X_train_scaled, y_train)

    # Train LDA
    model_lda = train_lda(X_train_scaled, y_train)

    # Train GPC
    model_gpc = train_gpc(X_train_pca, y_train)

    # 5. Inference & Evaluation
    print("\n--- Evaluating on Validation Set ---")

    # Get probabilities
    probs_lr = model_lr.predict_proba(X_val_scaled)
    probs_lda = model_lda.predict_proba(X_val_scaled)
    probs_gpc = model_gpc.predict_proba(X_val_pca)

    # Soft Voting Ensemble
    # Averaging probabilities is robust and well-calibrated if components are calibrated
    probs_ensemble = (probs_lr + probs_lda + probs_gpc) / 3.0

    # Calculate Metric
    # Ensure classes match. Sklearn classifiers sort classes alphabetically.
    # Since all models were trained on the same y_train, model.classes_ should be identical.
    assert np.array_equal(model_lr.classes_, model_lda.classes_)
    assert np.array_equal(model_lr.classes_, model_gpc.classes_)

    val_metric = log_loss(y_val, probs_ensemble)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(X_val, y_val, probs_ensemble, model_lr.classes_)

    # 7. Submission Logic
    if val_metric < VALIDATION_THRESHOLD:
        print(
            f"\nValidation metric {val_metric} is better than threshold {VALIDATION_THRESHOLD}."
        )
        print("Proceeding to generate submission with full dataset retraining...")

        # Load Combined Data (Train + Val)
        X_comb, y_comb, X_test, ids_test = get_combined_train_data(load_cached=True)

        # Preprocessing (Submission Phase)
        print("\n--- Preprocessing (Submission Phase) ---")

        # Pipeline A: Scaling
        X_comb_scaled, _, X_test_scaled = preprocess_data(
            X_comb, None, X_test, use_pca=False, cache_prefix="full", load_cached=True
        )

        # Pipeline B: Scaling + PCA
        X_comb_pca, _, X_test_pca = preprocess_data(
            X_comb, None, X_test, use_pca=True, cache_prefix="full", load_cached=True
        )

        # Training (Submission Phase)
        print("\n--- Retraining Models on Full Data ---")
        final_model_lr = train_logreg_cv(X_comb_scaled, y_comb)
        final_model_lda = train_lda(X_comb_scaled, y_comb)
        final_model_gpc = train_gpc(X_comb_pca, y_comb)

        # Inference (Test Set)
        print("\n--- Generating Test Predictions ---")
        test_probs_lr = final_model_lr.predict_proba(X_test_scaled)
        test_probs_lda = final_model_lda.predict_proba(X_test_scaled)
        test_probs_gpc = final_model_gpc.predict_proba(X_test_pca)

        # Ensemble
        test_probs_ensemble = (test_probs_lr + test_probs_lda + test_probs_gpc) / 3.0

        # Save Submission
        save_submission(
            ids_test,
            test_probs_ensemble,
            final_model_lr.classes_,
            SUBMISSION_OUTPUT_PATH,
        )

    else:
        print(
            f"\nValidation metric {val_metric} did not meet threshold {VALIDATION_THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
