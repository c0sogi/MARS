import os
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import RANDOM_SEED, SUBMISSION_FILE_PATH, CACHE_DIR
from library.data_loader import load_datasets
from library.models import tune_logistic_regression, get_hybrid_ensemble_components
from library.utils import calculate_log_loss, save_submission, clip_probabilities


def run_demo():
    # Set seeds for reproducibility
    np.random.seed(RANDOM_SEED)

    # Suppress warnings for cleaner output (e.g. from sklearn convergence)
    warnings.filterwarnings("ignore")

    print("=== Starting Library Demo Script ===")

    # -------------------------------------------------------------------------
    # 1. Data Loading (Split Mode for Validation)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Data (Split Mode)...")
    # We force load_cached_data=False to demonstrate the raw loading logic from metadata
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=False, combine_train_val=False
    )

    # Assertions to verify data loading
    assert X_train.shape[0] == len(y_train), "Train features and labels mismatch"
    assert X_val.shape[0] == len(y_val), "Val features and labels mismatch"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"
    print(f"  Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"  Number of classes: {len(classes)}")

    # -------------------------------------------------------------------------
    # 2. Hyperparameter Tuning
    # -------------------------------------------------------------------------
    print("\n[Step 2] Tuning Logistic Regression...")
    # This uses the grid defined in config.py (LR_C_GRID)
    optimal_c = tune_logistic_regression(X_train, y_train)

    assert optimal_c > 0, "Optimal C must be positive"
    print(f"  Selected Optimal C: {optimal_c}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Training (Split Mode)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Hybrid Ensemble on Split Data...")
    lda_model, bagging_model = get_hybrid_ensemble_components(optimal_c)

    # Train LDA (Generative Branch)
    print("  Fitting LDA...")
    lda_model.fit(X_train, y_train)

    # Train Bagging (Discriminative Branch)
    print("  Fitting Bagging Classifier...")
    bagging_model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 4. Evaluation on Validation Set
    # -------------------------------------------------------------------------
    print("\n[Step 4] Evaluating on Validation Set...")

    # Get probabilities from both models
    probs_lda = lda_model.predict_proba(X_val)
    probs_bagging = bagging_model.predict_proba(X_val)

    # Simple averaging ensemble
    probs_ensemble = (probs_lda + probs_bagging) / 2.0

    # Validate probabilities shape
    assert probs_ensemble.shape == (
        len(y_val),
        len(classes),
    ), "Probability shape mismatch"

    # Calculate Log Loss using the utility function
    val_loss = calculate_log_loss(y_val, probs_ensemble, class_labels=classes)
    print(f"  Validation Log Loss: {val_loss:.5f}")

    # Sanity check: Loss should be significantly better than random guessing
    # Random guess for 99 classes is approx ln(99) ~= 4.6
    assert val_loss < 4.0, f"Model performance ({val_loss}) is suspiciously poor."

    # -------------------------------------------------------------------------
    # 5. Final Training (Combined Mode) & Submission
    # -------------------------------------------------------------------------
    print("\n[Step 5] Retraining on Full Data and Generating Submission...")

    # Reload data in combined mode (merges Train + Val)
    # Note: X_val and y_val will be None in this mode
    X_train_full, y_train_full, _, _, X_test_final, test_ids_final, classes_final = (
        load_datasets(load_cached_data=False, combine_train_val=True)
    )

    assert len(X_train_full) > len(
        X_train
    ), "Combined dataset should be larger than split train"

    # Re-initialize models to reset them for full training
    lda_final, bagging_final = get_hybrid_ensemble_components(optimal_c)

    # Fit on full data
    print("  Fitting models on combined dataset...")
    lda_final.fit(X_train_full, y_train_full)
    bagging_final.fit(X_train_full, y_train_full)

    # Predict on Test
    print("  Predicting on Test set...")
    test_probs_lda = lda_final.predict_proba(X_test_final)
    test_probs_bagging = bagging_final.predict_proba(X_test_final)
    test_probs_ensemble = (test_probs_lda + test_probs_bagging) / 2.0

    # -------------------------------------------------------------------------
    # 6. Saving Submission
    # -------------------------------------------------------------------------
    print("\n[Step 6] Saving Submission...")

    # Use the utility function to save
    save_submission(
        ids=test_ids_final,
        class_names=classes_final,
        probs=test_probs_ensemble,
        output_path=SUBMISSION_FILE_PATH,
    )

    # Verify file creation
    assert os.path.exists(SUBMISSION_FILE_PATH), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(SUBMISSION_FILE_PATH)
    assert df_sub.shape == (
        len(test_ids_final),
        len(classes_final) + 1,
    ), "Submission CSV shape mismatch"
    assert "id" in df_sub.columns, "id column missing in submission"

    print(f"  Submission successfully saved to {SUBMISSION_FILE_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
