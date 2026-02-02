import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from provided library files
import importlib
import library.config

importlib.reload(library.config)
import library.utils

importlib.reload(library.utils)
import library.features

importlib.reload(library.features)
import library.transformers

importlib.reload(library.transformers)
import library.library

importlib.reload(library.library)
import library.selection

importlib.reload(library.selection)

from library.config import RANDOM_SEED, SUBMISSION_DIR, PROB_CLIP_EPS, FLOAT_PRECISION
from library.utils import set_seed, save_submission, clip_log_loss
from library.features import get_data
from library.library import get_expert_pool
from library.selection import run_selection, GreedySelector


def main():
    # 1. Setup Environment
    print("Initializing environment...")
    set_seed(RANDOM_SEED)

    # 2. Load Data
    # This handles metadata loading, image processing, and caching.
    # It returns DataFrames with original features + extracted morphometrics.
    print("\nLoading and processing data...")
    df_train, df_val, df_test = get_data(load_cached_data=True)

    # Basic Validation of Loaded Data
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    assert not df_train.isnull().values.any(), "Training data contains NaNs"
    assert (
        "hu_1" in df_train.columns
    ), "Morphometric features (hu_1) missing from dataframe"

    # Prepare Targets
    y_train = df_train["species"].values
    y_val = df_val["species"].values
    # Test set does not have species column

    # 3. Initialize Expert Pool
    # Retrieves the list of configured PFPGE experts (LDA variants)
    print("\nInitializing expert pool...")
    experts = get_expert_pool()
    print(f"Created {len(experts)} experts.")

    # 4. Train and Validate Experts
    print("\nTraining experts and generating validation predictions...")
    val_preds = {}
    test_preds = {}
    classes = None

    for i, expert in enumerate(experts):
        # Fit on training data
        # Expert handles column selection internally based on its config
        expert.fit(df_train, y_train)

        # Capture class names from the first expert (LDA sorts them alphabetically)
        if classes is None:
            classes = expert.estimator.classes_
        else:
            # Ensure all experts agree on class ordering
            assert np.array_equal(
                classes, expert.estimator.classes_
            ), f"Class mismatch in expert {expert.name}"

        # Predict on Validation set
        p_val = expert.predict_proba(df_val)
        val_preds[expert.name] = p_val

        # Predict on Test set
        p_test = expert.predict_proba(df_test)
        test_preds[expert.name] = p_test

        # Quick sanity check on probabilities
        assert np.allclose(
            p_val.sum(axis=1), 1.0
        ), f"Probabilities do not sum to 1 for {expert.name}"

        # Print performance of individual expert
        score = clip_log_loss(y_val, p_val)
        # print(f"  [{i+1}/{len(experts)}] {expert.name}: Log Loss = {score:.5f}")

    # 5. Ensemble Selection
    # Use Greedy Forward Selection to find optimal weights
    print("\nRunning Greedy Forward Selection...")
    # We limit iterations for the demo to ensure speed, though 50 is fast anyway
    weights, best_score = run_selection(
        val_preds, y_val, n_iterations=20, tolerance=1e-5
    )

    print(f"Best Ensemble Score: {best_score:.5f}")
    print("Ensemble Weights:", weights)

    # Verify that ensemble performs at least as well as the best single model (approx)
    single_best_score = min([clip_log_loss(y_val, p) for p in val_preds.values()])
    print(f"Single Best Model Score: {single_best_score:.5f}")

    # Allow for tiny floating point differences, but ensemble should generally be better or equal
    assert (
        best_score <= single_best_score + 1e-9
    ), "Ensemble selection failed to match or beat single best model."

    # 6. Generate Final Predictions
    print("\nGenerating final test predictions...")

    # Re-instantiate selector to use its predict_proba method easily
    # (run_selection is a helper that returns weights, but we can manually compute or use object)
    selector = GreedySelector()
    selector.weights_ = weights

    final_test_probs = selector.predict_proba(test_preds)

    # Verify final output shape and properties
    n_test_samples = len(df_test)
    n_classes = len(classes)

    assert final_test_probs.shape == (n_test_samples, n_classes)
    assert np.allclose(final_test_probs.sum(axis=1), 1.0)
    assert final_test_probs.dtype == FLOAT_PRECISION

    # 7. Save Submission
    print("\nSaving submission...")
    save_submission(df_test["id"].values, classes, final_test_probs)

    # Verify file creation
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify file content format
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (
        n_test_samples,
        n_classes + 1,
    ), "Submission has incorrect shape"
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert df_sub.iloc[0, 1:].sum() > 0.99, "Submission row sums seem incorrect"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
