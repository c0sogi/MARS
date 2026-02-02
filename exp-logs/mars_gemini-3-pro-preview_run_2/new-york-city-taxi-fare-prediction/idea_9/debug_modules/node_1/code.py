import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure reproducibility
np.random.seed(42)

# Import library modules
# We assume the script is running from the root directory where 'library' is a package
try:
    from library import config
    from library import data_loader
    from library import preprocessor
    from library import feature_engine
    from library import model_handler
except ImportError as e:
    print(f"Error importing library modules: {e}")
    sys.exit(1)


def main():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set a separate working directory for this demo to avoid cache conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config parameters
    config.WORKING_DIR = demo_working_dir
    config.DEBUG = True
    config.DEBUG_SIZE = 10_000  # Small subset for speed
    config.TRAIN_SUBSAMPLE_SIZE = 10_000

    # Override XGBoost params for rapid training
    config.XGB_PARAMS["n_estimators"] = 20
    config.XGB_PARAMS["learning_rate"] = 0.1
    config.XGB_PARAMS["max_depth"] = 4
    config.EARLY_STOPPING_ROUNDS = 5
    config.VERBOSE_EVAL = False

    # Update paths in config that depend on WORKING_DIR
    # (Note: In a real scenario, these might need re-initialization, but here we just patch the object)
    # The classes in the library re-construct paths using config.WORKING_DIR, so setting it is sufficient.

    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Debug Mode: {config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data Splits...")

    # We force load_cached_data=False to ensure we test the loading logic
    full_train, train_sub, val_df, test_df = data_loader.get_data_splits(
        load_cached_data=False
    )

    # Validation
    assert not full_train.empty, "Full training set should not be empty"
    assert not train_sub.empty, "Training subsample should not be empty"
    assert not val_df.empty, "Validation set should not be empty"
    assert not test_df.empty, "Test set should not be empty"

    print(f"    Full Train Shape: {full_train.shape}")
    print(f"    Train Subsample Shape: {train_sub.shape}")
    print(f"    Val Shape: {val_df.shape}")
    print(f"    Test Shape: {test_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------------------------------
    print("\n[3] Preprocessing Data (Clamping & Rounding)...")

    # We use the preprocessor module which handles caching internally.
    # Since we changed WORKING_DIR, it will process from scratch.
    full_train_proc, train_sub_proc, val_proc, test_proc = (
        preprocessor.get_preprocessed_splits(load_cached_data=False)
    )

    # Verify rounding logic (check if precision matches config)
    sample_lon = train_sub_proc["pickup_longitude"].iloc[0]
    # Check decimal places by string conversion logic or tolerance
    # config.COORD_PRECISION is 3
    assert (
        round(sample_lon, config.COORD_PRECISION) == sample_lon
    ), "Coordinates should be rounded to configured precision"

    print("    Preprocessing complete. Coordinates clamped and rounded.")

    # -------------------------------------------------------------------------
    # 4. Feature Engineering (Global Route Encoder)
    # -------------------------------------------------------------------------
    print("\n[4] Applying Global Route Encoder...")

    encoder = feature_engine.GlobalRouteEncoder()

    # Fit on the full training set (which is a subset in DEBUG mode)
    encoder.fit(full_train_proc, load_cached_data=False)

    # Transform Training Subsample (Vectorized K-Fold subtraction)
    train_sub_eng = encoder.transform_train_vectorized(train_sub_proc, num_folds=3)

    # Transform Validation and Test (Direct mapping)
    val_eng = encoder.transform_inference(val_proc)
    test_eng = encoder.transform_inference(test_proc)

    # Validation
    new_col = "route_avg_fare"
    assert (
        new_col in train_sub_eng.columns
    ), f"Feature {new_col} missing from train subsample"
    assert new_col in val_eng.columns, f"Feature {new_col} missing from validation set"
    assert new_col in test_eng.columns, f"Feature {new_col} missing from test set"

    # Check for NaNs in the new feature
    assert (
        not train_sub_eng[new_col].isnull().any()
    ), "NaNs found in engineered feature (Train)"
    assert not val_eng[new_col].isnull().any(), "NaNs found in engineered feature (Val)"

    print(f"    Feature '{new_col}' successfully created.")

    # -------------------------------------------------------------------------
    # 5. Model Training (XGBoost)
    # -------------------------------------------------------------------------
    print("\n[5] Training XGBoost Model...")

    handler = model_handler.ModelHandler()

    # Prepare targets
    y_train = train_sub_eng[config.TARGET_COL]
    y_val = val_eng[config.TARGET_COL]

    # Train
    model = handler.train_model(train_sub_eng, y_train, val_eng, y_val)

    # Validation
    assert model is not None, "Model object is None after training"
    assert os.path.exists(
        handler.model_path
    ), f"Model file not found at {handler.model_path}"

    print("    Model training complete and saved.")

    # -------------------------------------------------------------------------
    # 6. Prediction and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions and Submission...")

    # Generate predictions
    preds = handler.generate_predictions(test_eng)

    # Validate predictions
    assert len(preds) == len(test_eng), "Prediction count mismatch"
    assert (
        preds >= config.MIN_FARE
    ).all(), f"Predictions below minimum fare of {config.MIN_FARE}"

    # Create submission file
    handler.create_submission(test_df, preds)

    # Validate submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert len(sub_df) == len(test_df), "Submission row count incorrect"

    print(f"    Submission generated at {config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
