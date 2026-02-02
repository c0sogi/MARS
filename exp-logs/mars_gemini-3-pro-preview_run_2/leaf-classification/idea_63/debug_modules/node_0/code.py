import os
import sys
import numpy as np
import pandas as pd
import warnings
from library.utils import set_seed, save_submission, clipped_log_loss
from library.data_manager import load_data
from library.expert_library import generate_candidate_experts
from library.ensemble_selection import GreedySelector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Leaf Classification Demo ===")

    # 1. Setup
    # Set seed for reproducibility across numpy, torch, etc.
    set_seed(42)

    # 2. Load Data
    # load_data handles metadata reading and on-the-fly feature extraction (morphometrics)
    # It returns X (features) as DataFrames and y (targets) as encoded integers/arrays.
    print("\n[Step 1] Loading Data...")
    # We enable caching to speed up re-runs, though the first run will compute features.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        load_cached_data=True
    )

    # Verification of loaded data
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_val) > 0, "Validation data is empty"
    assert len(X_test) > 0, "Test data is empty"
    assert len(classes) == 99, f"Expected 99 classes, got {len(classes)}"
    print(f"Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Generate Experts
    # We generate a suite of pipelines based on the feature columns available.
    # The factory uses column names to group features (Shape, Margin, Texture, Morphometrics).
    print("\n[Step 2] Generating Candidate Experts...")
    feature_names = list(X_train.columns)
    experts = generate_candidate_experts(feature_names)
    print(f"Generated {len(experts)} experts.")

    # 4. Train Experts and Predict
    print("\n[Step 3] Training Experts and Generating Predictions...")
    val_preds = {}
    test_preds = {}

    # We will train all generated experts.
    # Since they are primarily LDA/PCA based (sklearn), they are computationally efficient.
    for i, expert in enumerate(experts):
        # Print progress every 5 experts to avoid clutter
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Training Expert {i+1}/{len(experts)}: {expert.name}")

        # Fit on training data
        # Note: The pipelines handle DataFrames via ColumnTransformer using the indices derived in expert_library
        expert.pipeline.fit(X_train, y_train)

        # Predict on validation set (for ensemble selection)
        p_val = expert.pipeline.predict_proba(X_val)
        val_preds[expert.name] = p_val

        # Predict on test set (for final submission)
        p_test = expert.pipeline.predict_proba(X_test)
        test_preds[expert.name] = p_test

        # Basic validation of predictions
        assert p_val.shape == (
            len(X_val),
            len(classes),
        ), "Validation prediction shape mismatch"
        assert not np.isnan(
            p_val
        ).any(), f"NaNs found in validation predictions of {expert.name}"

    # 5. Ensemble Selection
    # Use Greedy Forward Selection to find the best combination of experts
    # This optimizes the Multi-class Log Loss on the validation set.
    print("\n[Step 4] Optimizing Ensemble Weights...")
    selector = GreedySelector(max_iter=20, tolerance=1e-5)
    selector.fit(val_preds, y_val)

    selected_experts = selector.get_selected_experts()
    if not selected_experts:
        raise RuntimeError("Ensemble selection failed to select any experts.")

    # 6. Generate Final Predictions
    print("\n[Step 5] Generating Final Test Predictions...")
    # The selector computes the weighted average of the selected experts
    final_test_probs = selector.predict(test_preds)

    # Validate final probabilities
    assert final_test_probs.shape == (len(X_test), len(classes))
    # Check for valid probability range (allowing for small float errors)
    assert np.all(
        (final_test_probs >= 0) & (final_test_probs <= 1 + 1e-9)
    ), "Probabilities out of range"

    # 7. Save Submission
    submission_path = "./working/submission.csv"
    print(f"\n[Step 6] Saving submission to {submission_path}...")
    save_submission(test_ids, classes, final_test_probs, submission_path)

    # Final Verification
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created successfully. Shape: {df_sub.shape}")

        # Check first few columns to ensure format matches requirements
        expected_cols = ["id"] + list(classes)
        # We check set equality for columns because order might vary slightly if not strictly enforced,
        # but save_submission enforces order based on 'classes' list.
        assert (
            list(df_sub.columns) == expected_cols
        ), "Submission columns do not match requirements"
        assert len(df_sub) == len(X_test), "Submission row count mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
