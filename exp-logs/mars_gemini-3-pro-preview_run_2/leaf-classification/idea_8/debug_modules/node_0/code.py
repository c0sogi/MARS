import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from the provided library
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets, get_combined_train_data
from library.preprocessor import preprocess_data
from library.model_definitions import train_logreg_cv, train_lda, train_gpc
from library.config import SUBMISSION_DIR, WORKING_DIR


def run_pipeline_demonstration():
    print("Starting Pipeline Demonstration...")

    # 1. Setup and Reproducibility
    set_seed(42)

    # 2. Data Loading
    # We force load_cached=False to demonstrate the raw loading logic,
    # though caching is available in the library.
    print("\n--- Step 1: Loading Data ---")
    (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test) = (
        load_datasets(load_cached=False)
    )

    # Verification
    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )
    assert (
        X_train.shape[1] == 192
    ), "Feature count should be 192 (64 margin + 64 shape + 64 texture)"
    assert (
        len(y_train) == X_train.shape[0]
    ), "Mismatch between train features and labels"

    # 3. Preprocessing
    print("\n--- Step 2: Preprocessing ---")

    # 3a. Standard Scaling (for LogReg and LDA)
    # We use a unique cache prefix to avoid conflicts with other runs
    X_train_sc, X_val_sc, X_test_sc = preprocess_data(
        X_train,
        X_val,
        X_test,
        use_pca=False,
        cache_prefix="demo_run",
        load_cached=False,
    )

    # 3b. PCA (for GPC)
    # GPC scales cubically, so reducing dimensions helps, and it often performs better on dense features
    X_train_pca, X_val_pca, X_test_pca = preprocess_data(
        X_train, X_val, X_test, use_pca=True, cache_prefix="demo_run", load_cached=False
    )

    # Verification
    assert X_train_sc.shape[1] == 192, "Scaled data should retain original dimensions"
    assert X_train_pca.shape[1] < 192, "PCA data should have reduced dimensions"
    print(f"PCA reduced features from 192 to {X_train_pca.shape[1]}")

    # 4. Model Training and Evaluation
    print("\n--- Step 3: Model Training & Evaluation ---")

    # To ensure the demo runs quickly, we use the full dataset (N~700) which is small enough.
    # If the dataset were larger, we would subset here.

    # Model A: Logistic Regression (Discriminative Linear)
    # Uses Scaled Data
    model_lr = train_logreg_cv(X_train_sc, y_train)
    probs_val_lr = model_lr.predict_proba(X_val_sc)
    loss_lr = log_loss(y_val, probs_val_lr)
    print(f"Validation Log Loss (LogReg): {loss_lr:.4f}")

    # Model B: LDA (Generative Linear)
    # Uses Scaled Data
    model_lda = train_lda(X_train_sc, y_train)
    probs_val_lda = model_lda.predict_proba(X_val_sc)
    loss_lda = log_loss(y_val, probs_val_lda)
    print(f"Validation Log Loss (LDA): {loss_lda:.4f}")

    # Model C: Gaussian Process Classifier (Probabilistic Non-Linear)
    # Uses PCA Data
    model_gpc = train_gpc(X_train_pca, y_train)
    probs_val_gpc = model_gpc.predict_proba(X_val_pca)
    loss_gpc = log_loss(y_val, probs_val_gpc)
    print(f"Validation Log Loss (GPC): {loss_gpc:.4f}")

    # 5. Ensemble (Simple Average)
    print("\n--- Step 4: Ensembling ---")
    # Average the probabilities
    probs_val_ensemble = (probs_val_lr + probs_val_lda + probs_val_gpc) / 3.0
    loss_ensemble = log_loss(y_val, probs_val_ensemble)
    print(f"Validation Log Loss (Ensemble): {loss_ensemble:.4f}")

    # Verification of probabilities
    assert np.allclose(
        np.sum(probs_val_ensemble, axis=1), 1.0
    ), "Probabilities must sum to 1"

    # 6. Final Submission Generation
    print("\n--- Step 5: Generating Submission ---")

    # Generate predictions on Test set
    probs_test_lr = model_lr.predict_proba(X_test_sc)
    probs_test_lda = model_lda.predict_proba(X_test_sc)
    probs_test_gpc = model_gpc.predict_proba(X_test_pca)

    # Ensemble Test Predictions
    probs_test_final = (probs_test_lr + probs_test_lda + probs_test_gpc) / 3.0

    # Get class names from one of the models
    class_names = list(model_lr.classes_)

    # Define output path
    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Save
    save_submission(ids_test, probs_test_final, class_names, output_path)

    # Verify file creation
    assert os.path.exists(output_path), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(output_path)
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert len(df_sub) == len(ids_test), "Submission row count mismatch"
    # Check if all probability columns are present (excluding id)
    assert (
        len(df_sub.columns) - 1 == 99
    ), f"Expected 99 species columns, found {len(df_sub.columns) - 1}"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_pipeline_demonstration()
