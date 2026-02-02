import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing demonstration script...")

    # =========================================================================
    # 1. CONFIGURATION & OVERRIDES
    # =========================================================================
    # We import config first to override constants for a fast demo run.
    # This must be done before importing other modules that rely on these constants.
    import library.config as config

    print("Overriding configuration for speed...")
    # Reduce training rounds and early stopping for the demo
    config.MAX_ROUNDS = 10
    config.EARLY_STOPPING_ROUNDS = 5
    # Ensure working directory matches what we expect
    print(f"Working Directory: {config.WORKING_DIR}")

    # Now import the rest of the library modules
    from library.data_processor import load_and_prepare_data
    from library.ensemble_trainer import train_ensemble
    from library.inference_utils import soft_vote_predict, export_submission

    # Set global seed for reproducibility
    np.random.seed(config.SEED)

    # =========================================================================
    # 2. DATA LOADING & PROCESSING
    # =========================================================================
    print("\n--- Step 1: Data Loading & Processing ---")

    # We force reload (load_cached_data=False) to demonstrate the full pipeline,
    # but in a real scenario, True would be preferred.
    # Note: The provided function processes the full dataset.
    # Since we need to optimize for speed in this demo script, we will let it
    # process (which is reasonably fast for pandas) and then immediately subsample.
    X, y, X_test, test_ids = load_and_prepare_data(load_cached_data=False)

    print(f"Original Training Shape: {X.shape}")
    print(f"Original Test Shape: {X_test.shape}")

    # Subsample for rapid demonstration
    DEMO_TRAIN_SIZE = 2000
    DEMO_TEST_SIZE = 500

    print(
        f"Subsampling to {DEMO_TRAIN_SIZE} training samples and {DEMO_TEST_SIZE} test samples..."
    )
    X_subset = X.iloc[:DEMO_TRAIN_SIZE].copy()
    y_subset = y.iloc[:DEMO_TRAIN_SIZE].copy()
    X_test_subset = X_test.iloc[:DEMO_TEST_SIZE].copy()
    test_ids_subset = test_ids.iloc[:DEMO_TEST_SIZE].copy()

    # Encode targets to satisfy XGBoost requirements (Cite debug_lesson_3)
    le = LabelEncoder()
    y_subset = pd.Series(le.fit_transform(y_subset))

    # Verify subsampling
    assert len(X_subset) == DEMO_TRAIN_SIZE
    assert len(y_subset) == DEMO_TRAIN_SIZE

    # =========================================================================
    # 3. MODEL TRAINING
    # =========================================================================
    print("\n--- Step 2: Ensemble Training ---")

    # Use the parameters from config
    xgb_params = config.XGB_PARAMS.copy()

    # Adjust parameters for the small subset to avoid errors (e.g., num_class)
    # The dataset has classes 1-7. XGBoost expects 0-indexed labels for softprob usually,
    # or we ensure the num_class covers the max label.
    # Cover_Type is 1-7. num_class=8 covers indices 0-7.
    xgb_params["num_class"] = len(le.classes_)

    # Run training with 2 folds for speed
    N_FOLDS_DEMO = 2
    models, scores = train_ensemble(
        X_subset, y_subset, xgb_params, n_folds=N_FOLDS_DEMO
    )

    # Validation of training output
    assert (
        len(models) == N_FOLDS_DEMO
    ), f"Expected {N_FOLDS_DEMO} models, got {len(models)}"
    assert len(scores) == N_FOLDS_DEMO, "Scores list length mismatch"
    print(f"Training complete. Models trained: {len(models)}")

    # =========================================================================
    # 4. INFERENCE
    # =========================================================================
    print("\n--- Step 3: Inference ---")

    predictions = soft_vote_predict(models, X_test_subset)

    # Decode predictions back to original labels
    predictions = le.inverse_transform(predictions)

    # Validation of predictions
    assert len(predictions) == DEMO_TEST_SIZE, "Prediction count mismatch"
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"

    # Check if predictions are within valid range (1-7)
    # Note: XGBoost with num_class=8 outputs indices 0-7.
    # If the model learns correctly, it should predict 1-7.
    # Index 0 is unused if no label 0 exists in training.
    unique_preds = np.unique(predictions)
    print(f"Unique predicted classes: {unique_preds}")

    # =========================================================================
    # 5. SUBMISSION EXPORT
    # =========================================================================
    print("\n--- Step 4: Submission Export ---")

    # Define a demo output path
    demo_submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    export_submission(test_ids_subset, predictions, demo_submission_path)

    # Validation of export
    if os.path.exists(demo_submission_path):
        print(f"Submission file successfully created at {demo_submission_path}")

        # Verify content format
        df_sub = pd.read_csv(demo_submission_path)
        assert list(df_sub.columns) == [
            "Id",
            "Cover_Type",
        ], "Submission columns mismatch"
        assert len(df_sub) == DEMO_TEST_SIZE, "Submission row count mismatch"
        print("Submission file format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
