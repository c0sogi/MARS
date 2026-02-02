import os
import sys
import warnings
import numpy as np
import pandas as pd

# Import provided library modules
import library.config as conf
import library.utils as utils
import library.data_manager as dm
import library.expert_definitions as experts_lib
import library.ensemble_optimizer as optimizer


def main():
    # 1. Setup Environment
    # Set seeds for reproducibility
    utils.set_seed()
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting Leaf Classification Demonstration ===")

    # 2. Data Loading
    # The data_manager handles loading metadata, extracting image features (morphometrics),
    # combining them with tabular features, and caching the results.
    print("\n[Step 1] Loading and Processing Data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dm.load_data(
        load_cached_data=True
    )

    # Validate loaded data
    n_features_expected = 192 + 11  # 192 Tabular + 11 Morphometrics
    assert (
        X_train.shape[1] == n_features_expected
    ), f"Expected {n_features_expected} features, got {X_train.shape[1]}"
    assert (
        len(classes) == conf.N_CLASSES
    ), f"Expected {conf.N_CLASSES} classes, got {len(classes)}"
    print(
        f"Data Loaded Successfully. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
    )

    # 3. Expert Initialization
    print("\n[Step 2] Initializing Expert Pipelines...")
    experts = experts_lib.get_expert_library()
    print(f"Initialized {len(experts)} expert pipelines.")

    # 4. Training and Prediction
    print("\n[Step 3] Training Experts and Generating Predictions...")

    val_preds = {}
    test_preds = {}

    # Iterate over all defined experts
    for name, pipeline in experts.items():
        # Train the expert
        pipeline.fit(X_train, y_train)

        # Predict on Validation and Test sets
        # We need probabilities for Log Loss optimization
        p_val = pipeline.predict_proba(X_val)
        p_test = pipeline.predict_proba(X_test)

        # Store predictions
        val_preds[name] = p_val
        test_preds[name] = p_test

        # Basic validation of predictions
        assert not np.isnan(
            p_val
        ).any(), f"NaN values found in validation predictions for {name}"
        assert p_val.shape == (len(y_val), conf.N_CLASSES), f"Shape mismatch for {name}"

    print("Training complete.")

    # 5. Ensemble Optimization
    print("\n[Step 4] Optimizing Ensemble Weights (Greedy Forward Selection)...")
    # We use a reduced number of iterations for this demonstration to ensure speed,
    # though the full 50 iterations would also be quite fast.
    n_selection_iters = 20
    weights, best_score = optimizer.greedy_forward_selection(
        val_preds, y_val, n_iterations=n_selection_iters, with_replacement=True
    )

    print(f"Optimization Complete. Best Validation Log Loss: {best_score:.5f}")
    print(f"Ensemble Weights: {weights}")

    # 6. Submission Generation
    print("\n[Step 5] Generating Submission File...")

    # Compute weighted average of test predictions
    final_test_probs = np.zeros_like(list(test_preds.values())[0])
    total_weight = sum(weights.values())

    if total_weight == 0:
        raise ValueError("Total ensemble weight is zero. Selection failed.")

    for name, weight in weights.items():
        final_test_probs += test_preds[name] * weight

    final_test_probs /= total_weight

    # Ensure probabilities are clipped/normalized (though the scoring function does this,
    # it's good practice for the submission file to be clean)
    row_sums = final_test_probs.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    final_test_probs = final_test_probs / row_sums

    # Construct DataFrame
    submission_df = pd.DataFrame(final_test_probs, columns=classes)

    # Add 'id' column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Save to disk
    os.makedirs(conf.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(conf.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # Validate submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    # Check format against sample
    sample_df = pd.read_csv(conf.SAMPLE_SUBMISSION_PATH)
    assert (
        submission_df.shape[0] == sample_df.shape[0]
    ), "Submission row count mismatch."
    assert (
        submission_df.shape[1] == sample_df.shape[1]
    ), "Submission column count mismatch."
    # Check that columns match (ignoring order if necessary, but here we expect exact match)
    assert set(submission_df.columns) == set(
        sample_df.columns
    ), "Submission columns do not match sample."

    print(f"Submission saved successfully to: {submission_path}")
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
