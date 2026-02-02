import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library.config import RANDOM_SEED, SUBMISSION_PATH, WORKING_DIR
from library.data import load_data
from library.experts import build_expert_library
from library.ensemble import GreedySelector


def set_reproducibility(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    set_reproducibility(RANDOM_SEED)

    # Suppress warnings for cleaner output (e.g. sklearn convergence warnings, future warnings)
    warnings.filterwarnings("ignore")

    print("=== Leaf Classification Pipeline Demonstration ===")

    # 2. Data Loading and Feature Extraction
    print("\n[Step 1] Loading Data and Extracting Features...")
    # We set load_cached_data=False to demonstrate the feature extraction logic
    # (Physical features from images, Global features from CSV)
    # in a real run, True would be preferred for speed.
    data = load_data(load_cached_data=False)

    # Unpack the data dictionary
    X_train_global = data["X_train_global"]
    X_val_global = data["X_val_global"]
    X_test_global = data["X_test_global"]

    X_train_physical = data["X_train_physical"]
    X_val_physical = data["X_val_physical"]
    X_test_physical = data["X_test_physical"]

    y_train = data["y_train"]
    y_val = data["y_val"]

    test_ids = data["test_ids"]
    classes = data["classes"]

    # Validation of Data Shapes
    n_global_features = 192
    n_physical_features = 13

    assert (
        X_train_global.shape[1] == n_global_features
    ), f"Expected {n_global_features} global features, got {X_train_global.shape[1]}"
    assert (
        X_train_physical.shape[1] == n_physical_features
    ), f"Expected {n_physical_features} physical features, got {X_train_physical.shape[1]}"
    assert len(X_train_global) == len(
        y_train
    ), "Mismatch in training samples and labels"

    print(f"  Training Samples: {len(y_train)}")
    print(f"  Validation Samples: {len(y_val)}")
    print(f"  Test Samples: {len(test_ids)}")
    print(f"  Number of Classes: {len(classes)}")

    # 3. Expert Library Construction and Training
    print("\n[Step 2] Building and Training Experts...")
    experts = build_expert_library()
    print(f"  Initialized {len(experts)} experts.")

    val_preds_dict = {}
    test_preds_dict = {}

    for i, expert in enumerate(experts):
        # Determine which feature view this expert requires
        if expert.feature_type == "global":
            X_tr = X_train_global
            X_v = X_val_global
            X_te = X_test_global
        elif expert.feature_type == "physical":
            X_tr = X_train_physical
            X_v = X_val_physical
            X_te = X_test_physical
        else:
            raise ValueError(f"Unknown feature type: {expert.feature_type}")

        # Fit the expert
        # Note: The Expert class handles pipeline construction (Scaling -> LDA)
        expert.fit(X_tr, y_train)

        # Generate probabilities
        val_probs = expert.predict_proba(X_v)
        test_probs = expert.predict_proba(X_te)

        # Store for ensemble selection
        val_preds_dict[expert.name] = val_probs
        test_preds_dict[expert.name] = test_probs

        # Validate output probability shape and range
        assert val_probs.shape == (len(y_val), len(classes))
        assert test_probs.shape == (len(test_ids), len(classes))
        assert np.all(
            (val_probs >= 0) & (val_probs <= 1)
        ), "Probabilities out of range [0, 1]"

        # Print brief status (every few models to keep output concise)
        if (i + 1) % 2 == 0 or (i + 1) == len(experts):
            print(f"  Trained {i+1}/{len(experts)}: {expert.name}")

    # 4. Ensemble Selection
    print("\n[Step 3] Running Greedy Ensemble Selection...")
    selector = GreedySelector()

    # Fits the selector on validation data to find the best combination of experts
    selector.fit(val_preds_dict, y_val)

    selected_experts = selector.selected_experts
    print(f"  Selected {len(selected_experts)} experts for the final ensemble.")

    # Ensure at least one expert was selected
    if not selected_experts:
        raise RuntimeError("Ensemble selection failed to select any experts.")

    # 5. Inference on Test Set
    print("\n[Step 4] Generating Final Test Predictions...")
    # The selector aggregates predictions from the selected experts (averaging)
    final_test_probs = selector.predict(test_preds_dict)

    # 6. Submission File Generation
    print("\n[Step 5] Saving Submission...")

    # Construct DataFrame matching the sample_submission format
    submission_df = pd.DataFrame(final_test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save to disk
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    # Final Verification
    if os.path.exists(SUBMISSION_PATH):
        print(f"  Submission successfully saved to: {SUBMISSION_PATH}")

        # Quick check of the saved file
        df_check = pd.read_csv(SUBMISSION_PATH)
        expected_cols = len(classes) + 1
        if df_check.shape[1] != expected_cols:
            raise ValueError(
                f"Submission has {df_check.shape[1]} columns, expected {expected_cols}"
            )
        if df_check.shape[0] != len(test_ids):
            raise ValueError(
                f"Submission has {df_check.shape[0]} rows, expected {len(test_ids)}"
            )
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
