import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
import library.config as config
import library.image_processing as img_proc
import library.data_handler as data_handler
import library.model_zoo as model_zoo
import library.ensemble_selection as ensemble_sel

# Set seeds for reproducibility
np.random.seed(42)


def run_demo():
    print("Initializing Leaf Classification Demo...")

    # =========================================================================
    # 1. CONFIGURATION PATCHING (OPTIMIZE FOR SPEED)
    # =========================================================================
    # The requirement is to execute quickly. We enable DEBUG_MODE by patching
    # the imported variables in the library modules.
    print("Patching configuration for fast demonstration (DEBUG_MODE)...")

    # Patch config module
    config.DEBUG_MODE = True
    config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for demo

    # Patch image_processing module (since it imports variables directly)
    img_proc.DEBUG_MODE = True
    img_proc.DEBUG_SAMPLE_SIZE = 50

    # Ensure working directory is clean for a fresh run (optional, but good for demo)
    # We remove the specific cache file to force the ImageProcessor to run
    if os.path.exists(config.CACHE_TRAIN_FEATURES):
        os.remove(config.CACHE_TRAIN_FEATURES)
    if os.path.exists(config.CACHE_VAL_FEATURES):
        os.remove(config.CACHE_VAL_FEATURES)

    # =========================================================================
    # 2. DATA LOADING AND PROCESSING
    # =========================================================================
    print("\nLoading and processing data...")
    dm = data_handler.DataManager()

    # load_data triggers:
    # - Metadata loading
    # - Image processing (Monte-Carlo augmentation) -> extract_robust_morphometrics
    # - Feature merging and View creation (Global, Morph, Combined)
    # - Preprocessing (PowerTransformer)
    # We set load_cached_data=False to demonstrate the processing pipeline.
    (X_train_views, y_train, X_val_views, y_val, X_test_views, test_ids, classes) = (
        dm.load_data(load_cached_data=False)
    )

    # --- Verification ---
    print("Verifying data shapes...")
    # Check that we have the expected views
    expected_views = ["Global", "Morph", "Combined"]
    for view in expected_views:
        assert view in X_train_views, f"Missing view: {view}"
        assert view in X_val_views, f"Missing view: {view}"
        assert view in X_test_views, f"Missing view: {view}"

        # Verify sample counts match the debug size (or full size if debug failed)
        # Note: Train split is ~80% of total. If debug is 50 total rows from metadata,
        # the split happens before metadata saving.
        # The metadata files are fixed. Debug mode in ImageProcessor takes head(N).
        # So X_train should have min(len(train_csv), 50) rows.
        n_train = X_train_views[view].shape[0]
        assert (
            n_train <= 50
        ), f"Expected <= 50 training samples in debug mode, got {n_train}"

    assert (
        len(y_train) == X_train_views["Global"].shape[0]
    ), "Mismatch in X_train and y_train length"
    assert (
        len(y_val) == X_val_views["Global"].shape[0]
    ), "Mismatch in X_val and y_val length"

    print(f"Data loaded successfully. Classes: {len(classes)}")
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")

    # =========================================================================
    # 3. MODEL TRAINING (EXPERT LIBRARY)
    # =========================================================================
    print("\nTraining expert models...")
    experts = model_zoo.get_expert_library()

    # Dictionaries to store predictions
    val_preds = {}
    test_preds = {}

    # Limit number of experts for the demo speed if necessary,
    # but the models (LDA/QDA/NB) are very fast, so we run all.
    for i, exp in enumerate(experts):
        name = exp["name"]
        model = exp["model"]
        view = exp["view"]

        # print(f"Training {name} on {view} view...")

        # Fit model
        # Handle potential failures gracefully in a real loop, but here we expect success
        model.fit(X_train_views[view], y_train)

        # Predict
        p_val = model.predict_proba(X_val_views[view])
        p_test = model.predict_proba(X_test_views[view])

        # Store
        val_preds[name] = p_val
        test_preds[name] = p_test

        # --- Verification ---
        assert p_val.shape[1] == len(
            classes
        ), f"Model {name} output wrong number of classes"
        assert not np.isnan(p_val).any(), f"Model {name} produced NaNs in validation"

    print(f"Trained {len(experts)} expert models.")

    # =========================================================================
    # 4. ENSEMBLE SELECTION
    # =========================================================================
    print("\nRunning Greedy Forward Selection for Ensemble...")

    # Instantiate selector with reduced iterations for demo speed
    selector = ensemble_sel.GreedySelector(max_iter=10, tol=1e-4)

    # Fit selector on validation data
    selector.fit(val_preds, y_val)

    # Get learned weights
    weights = selector.get_weights()

    # --- Verification ---
    assert len(weights) > 0, "Ensemble selection failed to select any models"
    print(f"Selected {len(weights)} models for the final ensemble.")

    # =========================================================================
    # 5. INFERENCE AND SUBMISSION
    # =========================================================================
    print("\nGenerating final predictions...")

    # Compute weighted average of test predictions
    final_probs = ensemble_sel.predict_weighted(test_preds, weights)

    # --- Verification ---
    assert final_probs.shape == (
        len(test_ids),
        len(classes),
    ), "Final probability shape mismatch"
    assert np.all(
        (final_probs >= 0) & (final_probs <= 1)
    ), "Probabilities out of range [0, 1]"
    # Check row sums (should be approximately 1 due to normalization in predict_weighted)
    row_sums = final_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Verify against sample submission structure
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    # Check columns match
    assert list(submission_df.columns) == list(
        sample_sub.columns
    ), "Submission columns do not match sample"
    # Check ID count matches (in full run, here we might have fewer if debug mode affected test set loading)
    # Note: DataManager loads metadata.head(DEBUG_SIZE) in debug mode.
    # So test set size is also reduced.
    print(f"Submission generated with {len(submission_df)} rows.")


if __name__ == "__main__":
    run_demo()
