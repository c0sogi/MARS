import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import provided libraries
import library.config as config
import library.utils as utils
import library.data_processing as dp
import library.model_factory as mf
import library.stacking_trainer as st


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting Demonstration Script ===")

    # 1. Setup and Verification
    print("\n[1] Setting up environment...")
    utils.set_seed(42)
    has_gpu = utils.verify_gpu()
    if not has_gpu:
        print("Warning: GPU not detected. Training might be slow.")

    # 2. Demonstrate Data Processing Components
    print("\n[2] Demonstrating Feature Engineering...")
    # We load a very small chunk just to show the FeatureEngineer class works
    # We use the raw path directly to simulate raw input
    if os.path.exists(config.TRAIN_PATH):
        # Load first 100 rows
        raw_df = pd.read_csv(config.TRAIN_PATH, nrows=100)
        print(f"Loaded raw data sample with shape: {raw_df.shape}")

        fe = dp.FeatureEngineer()
        processed_df = fe.process(raw_df.copy())

        print(f"Processed data shape: {processed_df.shape}")

        # Validation
        # Check Reverse One-Hot
        if config.USE_REVERSE_ONE_HOT:
            assert (
                "Soil_Type" in processed_df.columns
            ), "Soil_Type column missing after processing"
            assert (
                "Wilderness_Area" in processed_df.columns
            ), "Wilderness_Area column missing after processing"
            # Check that binary columns are gone (Soil_Type1)
            # We check if any original soil columns are still present
            soil_cols_present = [
                c for c in config.SOIL_COLUMNS if c in processed_df.columns
            ]
            assert (
                len(soil_cols_present) == 0
            ), f"Binary Soil_Type columns were not dropped: {soil_cols_present}"

        # Check Geometric Features
        if config.USE_GEOMETRIC_FEATURES:
            assert (
                "Euclidean_Distance_To_Hydrology" in processed_df.columns
            ), "Geometric feature 'Euclidean_Distance_To_Hydrology' missing"
            assert (
                "Aspect_Sin" in processed_df.columns
            ), "Geometric feature 'Aspect_Sin' missing"

        print("Feature Engineering validation passed.")
    else:
        print(f"Error: {config.TRAIN_PATH} not found. Skipping data processing demo.")

    # 3. Demonstrate Model Wrapper
    print("\n[3] Demonstrating XGBWrapper...")
    # Create synthetic data for a quick test
    N_SAMPLES = 200
    N_FEATURES = 20
    X_syn = np.random.rand(N_SAMPLES, N_FEATURES)
    # Classes 1-7 (Target is 1-based in dataset)
    y_syn = np.random.randint(1, 8, N_SAMPLES)

    # Define params (reduced for demo)
    demo_params = config.L0_XGB_PARAMS.copy()
    demo_params["n_jobs"] = 4

    # Instantiate
    xgb_model = mf.XGBWrapper(
        params=demo_params, num_boost_round=5, early_stopping_rounds=2
    )

    # Fit (using same data for val to keep it simple)
    xgb_model.fit(X_syn, y_syn, X_syn, y_syn)

    # Predict
    preds = xgb_model.predict_proba(X_syn)

    # Validation
    assert preds.shape == (
        N_SAMPLES,
        config.NUM_CLASSES,
    ), f"Prediction shape mismatch: {preds.shape}"
    # Check probabilities sum to 1 (approx)
    assert np.allclose(
        preds.sum(axis=1), 1.0, atol=1e-5
    ), "Probabilities do not sum to 1"
    print("XGBWrapper validation passed.")

    # 4. Demonstrate Stacking Pipeline (Full Run)
    print("\n[4] Demonstrating StackingManager (Full Pipeline)...")

    # Patch parameters in stacking_trainer module to ensure quick execution
    # We modify the module attributes directly to override config defaults for this run
    st.N_FOLDS = 2
    st.L0_NUM_BOOST_ROUND = 10
    st.L0_EARLY_STOPPING_ROUNDS = 5
    st.L1_NUM_BOOST_ROUND = 10
    st.L1_EARLY_STOPPING_ROUNDS = 5

    # Instantiate Manager
    manager = st.StackingManager()

    # Run with debug sampling
    # We use 5000 samples to ensure enough class diversity for StratifiedKFold
    # load_cached_data=False forces the pipeline to process data and train models
    debug_size = 5000
    print(
        f"Running pipeline with debug_sample_size={debug_size}, folds={st.N_FOLDS}..."
    )

    try:
        manager.run(debug_sample_size=debug_size, load_cached_data=False)

        # 5. Verify Submission
        print("\n[5] Verifying Submission...")
        if os.path.exists(config.SUBMISSION_PATH):
            sub_df = pd.read_csv(config.SUBMISSION_PATH)
            print(f"Submission file loaded. Shape: {sub_df.shape}")

            # Check expected length (Test set has 400,000 rows)
            # Note: load_and_process does NOT subsample the test set, only train set.
            # So we expect full test set predictions.
            expected_len = 400000
            assert (
                len(sub_df) == expected_len
            ), f"Expected {expected_len} predictions, got {len(sub_df)}"

            # Check columns
            assert list(sub_df.columns) == [
                "Id",
                "Cover_Type",
            ], "Incorrect submission columns"

            # Check values
            assert (
                sub_df["Cover_Type"].dtype == np.int64
            ), "Cover_Type should be integer"
            assert (
                sub_df["Id"].iloc[0] == 814683 or sub_df["Id"].iloc[0] == 4000000
            ), "ID check failed (sanity check)"

            print("Submission validation passed.")
        else:
            raise FileNotFoundError(
                f"Submission file not found at {config.SUBMISSION_PATH}"
            )

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
