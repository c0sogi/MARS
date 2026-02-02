import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import is_classifier

# Import library modules
import library.config as config
import library.data_manager as data_manager
import library.models as models


def main():
    print("Starting Library Usage Demonstration...")

    # ==========================================================================
    # 1. OPTIMIZE CONFIGURATION FOR SPEED
    # ==========================================================================
    print("\n[Step 1] Optimizing hyperparameters for fast demonstration...")

    # Modify LR_PARAMS in config to speed up LogisticRegressionCV
    # We modify the dictionary in-place so changes reflect in library.models
    config.LR_PARAMS["cv"] = 2  # Reduce CV folds from 3 to 2
    config.LR_PARAMS["max_iter"] = 50  # Reduce max iterations
    config.LR_PARAMS["Cs"] = 2  # Reduce grid search granularity
    config.LR_PARAMS["n_jobs"] = 1  # Avoid multiprocessing overhead for small demo

    # Modify GPC_PARAMS
    config.GPC_PARAMS["n_jobs"] = 1

    print("Configuration updated.")

    # ==========================================================================
    # 2. DATA LOADING
    # ==========================================================================
    print("\n[Step 2] Loading and preprocessing data...")

    # Force reprocessing to verify logic (ignore cache)
    X_train, y_train, X_test, test_ids, classes = data_manager.load_and_preprocess_data(
        load_cached_data=False
    )

    # Verify data shapes
    n_classes = len(classes)
    print(
        f"Data loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
    )
    print(f"Test data: X_test shape: {X_test.shape}, Classes: {n_classes}")

    assert X_train.shape[0] == y_train.shape[0], "Mismatch in training samples/labels"
    assert X_train.shape[1] == X_test.shape[1], "Mismatch in feature dimensions"

    # Use full training data instead of subsampling to avoid class imbalance issues in CV
    # Cite debug_lesson_1
    print(
        f"Using full training set ({len(X_train)} samples) to ensure all classes are represented."
    )
    X_train_sub = X_train
    y_train_sub = y_train

    # ==========================================================================
    # 3. INDIVIDUAL MODEL DEMONSTRATION
    # ==========================================================================
    print("\n[Step 3] Verifying individual model components...")

    # --- A. Logistic Regression CV ---
    print("  -> Testing make_logistic_cv()...")
    lr_model = models.make_logistic_cv()
    assert is_classifier(lr_model), "make_logistic_cv did not return a classifier"

    lr_model.fit(X_train_sub, y_train_sub)
    lr_probs = lr_model.predict_proba(X_train_sub[:5])

    assert lr_probs.shape == (
        5,
        n_classes,
    ), f"LR output shape mismatch: {lr_probs.shape}"
    print("     Logistic Regression test passed.")

    # --- B. Linear Discriminant Analysis ---
    print("  -> Testing make_lda()...")
    lda_model = models.make_lda()

    lda_model.fit(X_train_sub, y_train_sub)
    lda_probs = lda_model.predict_proba(X_train_sub[:5])

    assert (
        lda_model.solver == config.LDA_PARAMS["solver"]
    ), "LDA solver parameter mismatch"
    assert lda_probs.shape == (
        5,
        n_classes,
    ), f"LDA output shape mismatch: {lda_probs.shape}"
    print("     LDA test passed.")

    # --- C. Gaussian Process Classifier Pipeline ---
    print("  -> Testing make_gpc_pipeline()...")
    gpc_pipeline = models.make_gpc_pipeline()

    # Check if pipeline has PCA step
    assert "pca" in gpc_pipeline.named_steps, "GPC pipeline missing PCA step"

    gpc_pipeline.fit(X_train_sub, y_train_sub)
    gpc_probs = gpc_pipeline.predict_proba(X_train_sub[:5])

    assert gpc_probs.shape == (
        5,
        n_classes,
    ), f"GPC output shape mismatch: {gpc_probs.shape}"
    print("     GPC Pipeline test passed.")

    # ==========================================================================
    # 4. ENSEMBLE DEMONSTRATION
    # ==========================================================================
    print("\n[Step 4] Verifying HybridEnsemble...")

    ensemble = models.HybridEnsemble()

    # Fit ensemble on subset
    ensemble.fit(X_train_sub, y_train_sub)

    # Check if internal models are fitted (by checking attributes usually set during fit)
    # Sklearn models usually have attributes ending in _ after fit
    assert hasattr(ensemble.lr, "classes_"), "Ensemble LR component not fitted"
    assert hasattr(ensemble.lda, "classes_"), "Ensemble LDA component not fitted"
    assert hasattr(ensemble.gpc, "classes_") or hasattr(
        ensemble.gpc.named_steps["gpc"], "classes_"
    ), "Ensemble GPC component not fitted"

    print("HybridEnsemble fitted successfully.")

    # ==========================================================================
    # 5. SUBMISSION GENERATION
    # ==========================================================================
    print("\n[Step 5] Generating submission for Test set...")

    # Predict on full test set
    test_probs = ensemble.predict_proba(X_test)

    # Verify probability constraints
    assert test_probs.shape == (
        len(X_test),
        n_classes,
    ), "Test probability shape mismatch"
    assert np.allclose(np.sum(test_probs, axis=1), 1.0), "Probabilities do not sum to 1"

    # Format submission DataFrame
    # Columns: id, <species_names...>
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Verify submission format
    expected_cols = ["id"] + list(classes)
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # Save submission
    submission_path = config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    # Final check of the file
    if os.path.exists(submission_path):
        saved_df = pd.read_csv(submission_path)
        print(f"Verification: Saved file has shape {saved_df.shape}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    main()
