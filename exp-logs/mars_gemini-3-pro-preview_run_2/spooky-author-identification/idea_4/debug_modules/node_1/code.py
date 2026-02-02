import sys
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from unittest.mock import MagicMock

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.models_classical as models_classical
import library.models_neural as models_neural
import library.stacking as stacking


def main():
    print("=== Starting Demonstration & Verification Script ===")

    # ==========================================
    # 1. Configuration & Patching for Speed
    # ==========================================
    print("\n[1] Patching configurations for rapid execution...")

    # Define a subset size for the demo
    SUBSET_SIZE = 60

    # Store original load_data to call it internally
    original_load_data = data_loader.load_data

    def mock_load_data():
        """
        Wrapper to load data and immediately subset it for speed.
        """
        train_df, val_df, test_df = original_load_data()

        # Subset to ensure speed
        train_subset = train_df.head(SUBSET_SIZE).copy()
        val_subset = val_df.head(SUBSET_SIZE).copy()
        test_subset = test_df.head(SUBSET_SIZE).copy()

        return train_subset, val_subset, test_subset

    # Apply the mock to all modules that use load_data
    data_loader.load_data = mock_load_data
    models_classical.load_data = mock_load_data
    models_neural.load_data = mock_load_data
    stacking.load_data = mock_load_data

    # Override constants in modules to run faster
    # 2 Folds is the minimum for CV
    config.N_FOLDS = 2
    models_classical.N_FOLDS = 2
    models_neural.N_FOLDS = 2

    # Reduce SVD components
    data_loader.SVD_COMPONENTS = 5

    # Reduce Neural Training params
    models_neural.EPOCHS = 1
    models_neural.BATCH_SIZE = 4
    models_neural.MAX_LENGTH = 16  # Short sequence length for speed

    # Force recalculation by ignoring cache for this run
    # We do this by modifying the behavior of the functions or just passing flags where available.
    # The provided functions accept `load_cached_data` or `load_cached_preds`.

    print("Configuration patched successfully.")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test clip_probabilities
    probs = np.array([-0.1, 0.5, 1.2, 0.0, 1.0])
    clipped = utils.clip_probabilities(probs)
    epsilon = 1e-15
    assert (clipped >= epsilon).all() and (
        clipped <= 1 - epsilon
    ).all(), "Clipping failed boundaries"
    assert np.allclose(clipped[1], 0.5), "Clipping altered valid probability"
    print(" - clip_probabilities: OK")

    # Test calculate_log_loss
    y_true = np.array([0, 1, 2])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    loss = utils.calculate_log_loss(y_true, y_pred)
    assert (
        loss < 0.6
    ), f"Log loss calculation seems incorrect (too high for good preds): {loss}"
    print(f" - calculate_log_loss: OK ({loss:.4f})")

    # ==========================================
    # 3. Verify Data Loader & Feature Engineering
    # ==========================================
    print("\n[3] Verifying Data Loader & Feature Engineering...")

    # Test Data Loading
    train_df, val_df, test_df = data_loader.load_data()
    assert len(train_df) == SUBSET_SIZE
    assert len(val_df) == SUBSET_SIZE
    assert len(test_df) == SUBSET_SIZE
    print(" - Data Loading (Subset): OK")

    # Test TF-IDF
    # We force load_cached_data=False to ensure code runs
    train_tfidf, val_tfidf, test_tfidf = data_loader.get_tfidf_features(
        train_df["text"], val_df["text"], test_df["text"], load_cached_data=False
    )
    assert sp.issparse(train_tfidf)
    assert train_tfidf.shape[0] == SUBSET_SIZE
    print(f" - TF-IDF Generation: OK (Shape: {train_tfidf.shape})")

    # Test SVD
    train_svd, val_svd, test_svd = data_loader.get_svd_features(
        train_tfidf, val_tfidf, test_tfidf, load_cached_data=False
    )
    assert train_svd.shape == (SUBSET_SIZE, data_loader.SVD_COMPONENTS)
    print(f" - SVD Generation: OK (Shape: {train_svd.shape})")

    # ==========================================
    # 4. Verify Classical Models
    # ==========================================
    print("\n[4] Verifying Classical Models (LR, NB, XGB)...")

    # Run CV
    classical_results = models_classical.run_classical_cv(load_cached_preds=False)

    # Check keys
    expected_keys = ["lr_oof", "lr_test", "nb_oof", "nb_test", "xgb_oof", "xgb_test"]
    for k in expected_keys:
        assert k in classical_results, f"Missing key in classical results: {k}"

    # Check shapes
    # Note: OOF size is len(train) + len(val) because the code concatenates them for CV
    total_train_samples = len(train_df) + len(val_df)
    assert classical_results["lr_oof"].shape == (total_train_samples, 3)
    assert classical_results["lr_test"].shape == (len(test_df), 3)

    print(" - Classical Models CV: OK")

    # ==========================================
    # 5. Verify Neural Models
    # ==========================================
    print("\n[5] Verifying Neural Models (DeBERTa, RoBERTa)...")
    print("    (This may take a moment, running 1 epoch on small subset)")

    # Run CV
    neural_results = models_neural.run_neural_cv(load_cached_preds=False)

    # Check keys
    expected_neural_keys = [
        "deberta_oof",
        "deberta_test",
        "roberta_oof",
        "roberta_test",
    ]
    for k in expected_neural_keys:
        assert k in neural_results, f"Missing key in neural results: {k}"

    # Check shapes
    assert neural_results["deberta_oof"].shape == (total_train_samples, 3)
    assert neural_results["deberta_test"].shape == (len(test_df), 3)

    print(" - Neural Models CV: OK")

    # ==========================================
    # 6. Verify Stacking (Meta-Learner)
    # ==========================================
    print("\n[6] Verifying Stacking & Submission...")

    # Combine results
    all_oof = {**classical_results, **neural_results}
    # Filter only test keys for the second argument
    all_test = {
        k: v for k, v in {**classical_results, **neural_results}.items() if "test" in k
    }

    # Run Stacking
    stacking.train_meta_learner(all_oof, all_test)

    # Check Submission File
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    assert len(sub_df) == len(
        test_df
    ), f"Submission length mismatch. Expected {len(test_df)}, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Submission columns mismatch"

    # Check values are probabilities
    probs = sub_df[["EAP", "HPL", "MWS"]].values
    # Allow small float tolerance, but they should be roughly summing to 1 (softmax/predict_proba)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print(f" - Submission Generated: OK ({submission_path})")
    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    # Ensure reproducibility
    utils.seed_everything(42)
    main()
