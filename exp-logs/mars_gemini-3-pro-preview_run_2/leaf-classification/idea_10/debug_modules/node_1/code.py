import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.data_processing import process_data
from library.model_factory import get_logistic_cv, get_lda, train_and_evaluate
from library.ensemble_strategy import SelectiveEnsemble, generate_submission
from library.config import FEATURE_VIEWS, ALL_VIEWS, RANDOM_SEED, SUBMISSION_FILE_PATH


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting demonstration of Leaf Classification Pipeline...")
    set_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 1. Data Processing
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Data Processing ---")

    # We use a debug_size to keep the execution fast for this demonstration.
    # load_cached_data=False ensures we demonstrate the full processing logic.
    debug_limit = None
    print(f"Loading and processing data (Debug Limit: {debug_limit} samples)...")

    X_train_views, y_train, X_test_views, test_ids, classes = process_data(
        load_cached_data=False, debug_size=debug_limit
    )

    # Verification: Check data structures
    print("Verifying data structures...")

    # Check that we have all expected views
    expected_views = ["Global"] + list(FEATURE_VIEWS.keys())
    for view in expected_views:
        assert view in X_train_views, f"Missing view '{view}' in training data"
        assert view in X_test_views, f"Missing view '{view}' in test data"

        # Check shapes
        n_train = X_train_views[view].shape[0]
        n_test = X_test_views[view].shape[0]
        n_features = X_train_views[view].shape[1]

        assert n_train == len(y_train), f"Mismatch in training samples for {view}"
        assert n_test == len(test_ids), f"Mismatch in test samples for {view}"
        assert n_features > 0, f"No features found for {view}"

    print(f"Data loaded successfully. Classes: {len(classes)}")
    print(f"Training samples: {len(y_train)}, Test samples: {len(test_ids)}")

    # -------------------------------------------------------------------------
    # 2. Model Training and Ensemble Building
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Model Training & Ensemble Building ---")

    # Initialize the ensemble strategy
    # Tolerance of 0.1 means we accept models within 0.1 log loss of the best Global model
    ensemble = SelectiveEnsemble(tolerance=0.1)

    # Iterate over views and train models
    for view_name in ALL_VIEWS:
        print(f"\nProcessing View: {view_name}")

        if view_name not in X_train_views:
            print(f"Skipping {view_name} (not found in data)")
            continue

        X_view = X_train_views[view_name]

        # --- Model A: Logistic Regression ---
        lr_model = get_logistic_cv()
        lr_name = f"LR_{view_name}"

        # Train and get CV score
        # Note: train_and_evaluate handles fitting and scoring
        fitted_lr, lr_score = train_and_evaluate(
            lr_model, X_view, y_train, model_name=lr_name
        )

        # Add to ensemble candidates
        ensemble.add_candidate(fitted_lr, view_name, lr_score, lr_name)

        # --- Model B: LDA ---
        lda_model = get_lda()
        lda_name = f"LDA_{view_name}"

        # Train and get CV score
        fitted_lda, lda_score = train_and_evaluate(
            lda_model, X_view, y_train, model_name=lda_name
        )

        # Add to ensemble candidates
        ensemble.add_candidate(fitted_lda, view_name, lda_score, lda_name)

    # -------------------------------------------------------------------------
    # 3. Ensemble Optimization and Prediction
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Ensemble Optimization ---")

    # Filter candidates based on the global baseline
    ensemble.optimize_selection()

    # Check if we have selected candidates
    if not ensemble.selected_candidates:
        raise RuntimeError("Ensemble selection failed to select any models.")

    print(f"Selected {len(ensemble.selected_candidates)} models for final prediction.")

    print("\nGenerating predictions on test set...")
    predictions = ensemble.predict(X_test_views)

    # Verification: Check prediction shape and values
    assert predictions.shape == (
        len(test_ids),
        len(classes),
    ), f"Prediction shape mismatch. Expected {(len(test_ids), len(classes))}, got {predictions.shape}"

    # Check probabilities are valid (0-1 range)
    assert np.all(predictions >= 0) and np.all(
        predictions <= 1
    ), "Predictions contain values outside [0, 1]"

    # Check rows sum to approximately 1
    row_sums = np.sum(predictions, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Submission Generation ---")

    generate_submission(
        predictions, test_ids, classes, output_path=SUBMISSION_FILE_PATH
    )

    # Verification: Check file content
    assert os.path.exists(SUBMISSION_FILE_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_FILE_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Verify columns
    expected_cols = ["id"] + list(classes)
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match expected classes"

    # Verify IDs match
    assert np.array_equal(
        df_sub["id"].values, test_ids
    ), "Submission IDs do not match test IDs"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
