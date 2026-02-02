import os
import sys
import warnings
import numpy as np
import pandas as pd

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import setup_directories, INPUT_DIR, METADATA_DIR, SUBMISSION_FILE
from library.utils import seed_everything, clipped_log_loss, save_submission
from library.image_processing import extract_single_image_features
from library.data_manager import load_dataset
from library.expert_zoo import generate_expert_library
from library.ensemble_selection import GreedySelector


def main():
    print("Initializing Demonstration...")

    # 1. Setup Environment
    # -------------------------------------------------------------------------
    setup_directories()
    seed_everything(42)
    print("Environment setup complete.")

    # 2. Verify Image Processing Logic
    # -------------------------------------------------------------------------
    print("\nVerifying Image Processing...")
    # Load train metadata to find a valid image path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_meta = pd.read_csv(train_meta_path)
        # Construct full path for the first image
        # Metadata image_path is relative, e.g., "images/12.jpg"
        first_img_rel_path = df_meta.iloc[0]["image_path"]
        first_img_path = os.path.join(INPUT_DIR, first_img_rel_path)

        if os.path.exists(first_img_path):
            features = extract_single_image_features(first_img_path)

            # Verify feature vector shape (11 features expected: 7 Hu + 4 Geometric)
            assert features.shape == (
                11,
            ), f"Expected 11 morphometric features, got {features.shape[0]}"

            # Verify values are not all zero (unless image is empty/invalid)
            if np.sum(features) == 0:
                print(
                    "Warning: Extracted features are all zero. Image might be invalid."
                )
            else:
                print(f"Successfully extracted 11 features from {first_img_rel_path}")
        else:
            print(f"Image file not found: {first_img_path}. Skipping image check.")
    else:
        print("Metadata file not found. Skipping image check.")

    # 3. Load Dataset
    # -------------------------------------------------------------------------
    print("\nLoading Dataset...")
    # load_dataset handles merging tabular features with extracted morphometrics
    # and caching the result in ./working/cache
    data = load_dataset(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]
    feature_groups = data["feature_groups"]

    # Basic assertions to ensure data integrity
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature count mismatch"
    assert len(X_train) == len(y_train), "Train features/labels length mismatch"
    print(f"Training Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Feature Count: {X_train.shape[1]}")
    print(f"Classes: {len(classes)}")

    # 4. Generate and Train Experts
    # -------------------------------------------------------------------------
    print("\nGenerating and Training Experts...")

    # Generate the library of experts based on the FR-SPPE strategy
    experts = generate_expert_library(feature_groups)
    print(f"Generated {len(experts)} experts defined in the zoo.")

    val_predictions = {}
    test_predictions = {}

    # Train each expert
    # Note: For this demo, we use the full training set provided by load_dataset.
    # The dataset is small enough that LDA/QDA fits very quickly.
    for i, expert in enumerate(experts):
        # Fit
        expert.fit(X_train, y_train)

        # Predict Probabilities
        p_val = expert.predict_proba(X_val)
        p_test = expert.predict_proba(X_test)

        # Store
        val_predictions[expert.name] = p_val
        test_predictions[expert.name] = p_test

        # Validation check for probability shape
        assert p_val.shape == (
            len(X_val),
            len(classes),
        ), f"Expert {expert.name} output shape mismatch"

    print("All experts trained successfully.")

    # 5. Ensemble Selection (Greedy Forward Selection)
    # -------------------------------------------------------------------------
    print("\nOptimizing Ensemble...")

    # Instantiate selector with limited steps for demonstration speed
    # In a full run, max_steps could be higher (e.g., 100-200)
    selector = GreedySelector(max_steps=20, tolerance=1e-5)

    # Fit selector on validation data
    selector.fit(val_predictions, y_val)

    # Verify selection
    if not selector.selected_experts:
        raise RuntimeError("GreedySelector failed to select any experts.")

    print(f"Best Validation Loss: {selector.best_loss:.5f}")

    # 6. Generate Test Predictions
    # -------------------------------------------------------------------------
    print("\nGenerating Final Predictions...")

    # Compute weighted average of test predictions using selected experts
    final_test_probs = selector.predict(test_predictions)

    # Verify output shape and constraints
    assert final_test_probs.shape == (
        len(X_test),
        len(classes),
    ), "Final prediction shape mismatch"
    assert np.all(final_test_probs >= 0) and np.all(
        final_test_probs <= 1
    ), "Probabilities out of bounds"

    # 7. Create Submission
    # -------------------------------------------------------------------------
    print("\nSaving Submission...")
    save_submission(test_ids, final_test_probs, classes, output_path=SUBMISSION_FILE)

    # Verify file creation
    if os.path.exists(SUBMISSION_FILE):
        df_sub = pd.read_csv(SUBMISSION_FILE)
        print(f"Submission file created at {SUBMISSION_FILE}")
        print(f"Submission shape: {df_sub.shape}")

        # Validate submission format
        assert df_sub.shape[0] == len(test_ids), "Submission row count mismatch"
        assert df_sub.shape[1] == len(classes) + 1, "Submission column count mismatch"
        assert "id" in df_sub.columns, "id column missing"
    else:
        raise FileNotFoundError("Submission file was not saved.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
