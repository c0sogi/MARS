import os
import pandas as pd
import numpy as np
import library.config as config
from library.utils import set_seed, save_submission
from library.data_loader import load_and_merge_data
from library.feature_engineering import FeaturePreprocessor, get_processed_data
from library.model import PizzaRandomForest


def main():
    print("Initializing demonstration...")
    # 1. Set Random Seed for Reproducibility
    set_seed(42)

    # =========================================================================
    # DEMO 1: Data Loading
    # =========================================================================
    print("\n--- Demo 1: Data Loading (Debug Mode) ---")
    # Load a small subset of data using debug=True
    train_df, val_df, test_df = load_and_merge_data(debug=True, debug_size=50)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Assertions to verify data loading
    assert not train_df.empty, "Training dataframe should not be empty"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"
    assert "request_id" in test_df.columns, "ID column missing in test"

    # =========================================================================
    # DEMO 2: Feature Engineering
    # =========================================================================
    print("\n--- Demo 2: Feature Engineering ---")
    preprocessor = FeaturePreprocessor()

    # Fit on training data
    preprocessor.fit(train_df)

    # Transform all splits
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    print(f"Processed Train Features shape: {X_train.shape}")

    # Verify Feature Structure
    # Expected columns: Numeric columns defined in config + Hashed text columns
    expected_numeric_count = len(
        [c for c in config.NUMERIC_COLS if c in train_df.columns]
    )
    expected_text_count = config.HASH_VECTOR_SIZE
    expected_total_cols = expected_numeric_count + expected_text_count

    assert (
        X_train.shape[1] == expected_total_cols
    ), f"Expected {expected_total_cols} features, got {X_train.shape[1]}"

    # Verify no missing values after processing (Imputer should handle them)
    assert not X_train.isnull().any().any(), "Processed features contain NaNs"
    assert not X_test.isnull().any().any(), "Processed test features contain NaNs"

    # Prepare targets
    y_train = train_df[config.TARGET_COL].astype(int)
    y_val = val_df[config.TARGET_COL].astype(int)

    # =========================================================================
    # DEMO 3: Model Training
    # =========================================================================
    print("\n--- Demo 3: Model Training ---")

    # Optimize hyperparameters for speed in this demo
    # We modify the configuration dictionary in memory before instantiating the model
    original_n_estimators = config.RF_PARAMS["n_estimators"]
    config.RF_PARAMS["n_estimators"] = 10  # Reduce for speed
    config.RF_PARAMS["max_depth"] = 5  # Reduce for speed

    print(f"Modified RF Params for demo: {config.RF_PARAMS}")

    model = PizzaRandomForest()
    model.train(X_train, y_train, X_val, y_val)

    # Restore config (good practice, though script ends shortly)
    config.RF_PARAMS["n_estimators"] = original_n_estimators

    # =========================================================================
    # DEMO 4: Prediction and Submission
    # =========================================================================
    print("\n--- Demo 4: Prediction and Submission ---")

    # Generate predictions on test set
    test_probs = model.predict_proba(X_test)

    # Verify predictions
    assert len(test_probs) == len(test_df), "Mismatch in prediction length"
    assert np.all(
        (test_probs >= 0) & (test_probs <= 1)
    ), "Probabilities out of [0, 1] range"

    print(f"Generated {len(test_probs)} predictions.")
    print(f"Sample predictions: {test_probs[:5]}")

    # Save submission
    # We use a temporary path for the demo to avoid overwriting main submission if needed,
    # or just use the default one. Here we use a specific demo file.
    demo_submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    save_submission(
        request_ids=test_df["request_id"],
        probabilities=test_probs,
        output_path=demo_submission_path,
    )

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not created"

    df_sub = pd.read_csv(demo_submission_path)
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect submission columns"
    assert len(df_sub) == len(test_df), "Incorrect number of rows in submission"

    # =========================================================================
    # DEMO 5: Pipeline Integration Wrapper
    # =========================================================================
    print("\n--- Demo 5: Pipeline Wrapper (get_processed_data) ---")
    # This function handles loading, processing, and caching automatically.
    # We run it in debug mode to ensure it works without consuming too much time.

    X_tr, y_tr, X_v, y_v, X_te, te_ids = get_processed_data(
        load_cached_data=False, debug=True  # Force re-compute to test logic
    )

    assert X_tr.shape[0] == 50, "Debug mode did not return correct sample size"
    assert len(te_ids) == 50, "Debug mode test IDs length incorrect"

    print("Pipeline wrapper executed successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
