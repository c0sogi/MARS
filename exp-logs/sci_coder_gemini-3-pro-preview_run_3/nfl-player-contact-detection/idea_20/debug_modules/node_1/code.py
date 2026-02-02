import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb

# Import library modules
import library.config as C
import library.utils as U
import library.data_manager as DM
import library.features_stream_a as FA
import library.features_stream_b as FB
import library.trainer as TR
import library.inference as INF


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> 1. Setting up configuration for demo run...")

    # Set random seed for reproducibility
    np.random.seed(C.SEED)

    # Monkey-patch configuration for speed
    # Reduce estimators to ensure training finishes quickly in this demo
    C.STREAM_A_PARAMS["n_estimators"] = 10
    C.STREAM_A_PARAMS["early_stopping_rounds"] = 5
    C.STREAM_B_PARAMS["n_estimators"] = 10
    C.STREAM_B_PARAMS["early_stopping_rounds"] = 5
    C.EARLY_STOPPING_ROUNDS = 5

    # Define custom modes for caching to avoid overwriting production cache
    # This ensures we use separate cache files for this debug run
    MODE_TRAIN = "train_demo"
    MODE_VAL = "val_demo"
    MODE_TEST = "test_demo"

    # -------------------------------------------------------------------------
    # 2. Data Loading (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n>>> 2. Loading data (Debug Mode)...")

    # Load Train Data (Sampled)
    df_train, df_helmets_train = DM.load_and_merge_data(mode="train", debug=True)

    # Load Validation Data (Sampled)
    df_val, df_helmets_val = DM.load_and_merge_data(mode="validation", debug=True)

    # Load Test Data (Sampled)
    df_test, df_helmets_test = DM.load_and_merge_data(mode="test", debug=True)

    # Verify data loaded
    assert not df_train.empty, "Training dataframe is empty"
    assert not df_val.empty, "Validation dataframe is empty"
    assert not df_test.empty, "Test dataframe is empty"
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # -------------------------------------------------------------------------
    # 3. Stream Splitting
    # -------------------------------------------------------------------------
    print(
        "\n>>> 3. Splitting data into Stream A (Player-Player) and Stream B (Player-Ground)..."
    )

    train_a, train_b = DM.split_by_stream(df_train)
    val_a, val_b = DM.split_by_stream(df_val)
    test_a, test_b = DM.split_by_stream(df_test)

    print(f"Train Stream A: {len(train_a)}, Stream B: {len(train_b)}")

    # -------------------------------------------------------------------------
    # 4. Feature Generation
    # -------------------------------------------------------------------------
    print("\n>>> 4. Generating Features...")

    # --- Stream A (Player-Player Interaction) ---
    print("Generating Stream A features...")
    # Note: passing custom mode strings to create separate cache files
    X_train_a, y_train_a, ids_train_a = FA.generate_stream_a_features(
        train_a, df_helmets_train, mode=MODE_TRAIN, load_cached_data=False
    )
    X_val_a, y_val_a, ids_val_a = FA.generate_stream_a_features(
        val_a, df_helmets_val, mode=MODE_VAL, load_cached_data=False
    )
    X_test_a, y_test_a, ids_test_a = FA.generate_stream_a_features(
        test_a, df_helmets_test, mode=MODE_TEST, load_cached_data=False
    )

    # Verify Stream A features
    assert X_train_a.shape[0] == len(y_train_a), "Mismatch in X and y for Train A"
    assert X_train_a.shape[1] > 0, "No features generated for Stream A"

    # --- Stream B (Player-Ground Impact) ---
    print("Generating Stream B features...")
    X_train_b, y_train_b, ids_train_b = FB.generate_stream_b_features(
        train_b, mode=MODE_TRAIN, load_cached_data=False
    )
    X_val_b, y_val_b, ids_val_b = FB.generate_stream_b_features(
        val_b, mode=MODE_VAL, load_cached_data=False
    )
    X_test_b, y_test_b, ids_test_b = FB.generate_stream_b_features(
        test_b, mode=MODE_TEST, load_cached_data=False
    )

    # Verify Stream B features
    assert X_train_b.shape[0] == len(y_train_b), "Mismatch in X and y for Train B"
    assert X_train_b.shape[1] > 0, "No features generated for Stream B"

    # -------------------------------------------------------------------------
    # 5. Model Training
    # -------------------------------------------------------------------------
    print("\n>>> 5. Training Models...")

    # Train Stream A Model
    model_a, thresh_a, mcc_a = TR.train_stream_model(
        stream_type="A",
        X_train=X_train_a,
        y_train=y_train_a,
        X_val=X_val_a,
        y_val=y_val_a,
        save_model=True,
    )

    # Train Stream B Model
    model_b, thresh_b, mcc_b = TR.train_stream_model(
        stream_type="B",
        X_train=X_train_b,
        y_train=y_train_b,
        X_val=X_val_b,
        y_val=y_val_b,
        save_model=True,
    )

    # Validate models were created
    assert isinstance(model_a, xgb.XGBClassifier), "Model A is not an XGBClassifier"
    assert isinstance(model_b, xgb.XGBClassifier), "Model B is not an XGBClassifier"
    assert os.path.exists(
        os.path.join(C.WORKING_DIR, "model_stream_a.json")
    ), "Model A file not found"
    assert os.path.exists(
        os.path.join(C.WORKING_DIR, "model_stream_b.json")
    ), "Model B file not found"

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> 6. Running Inference Pipeline...")

    # The pipeline loads the saved models, optimizes thresholds on validation data,
    # predicts on test data, and generates the submission file.
    INF.run_inference_pipeline(
        X_val_a=X_val_a,
        y_val_a=y_val_a,
        X_val_b=X_val_b,
        y_val_b=y_val_b,
        X_test_a=X_test_a,
        ids_test_a=ids_test_a,
        X_test_b=X_test_b,
        ids_test_b=ids_test_b,
    )

    # -------------------------------------------------------------------------
    # 7. Final Verification
    # -------------------------------------------------------------------------
    print("\n>>> 7. Verifying Submission...")

    if os.path.exists(C.SUBMISSION_PATH):
        df_sub = pd.read_csv(C.SUBMISSION_PATH)
        print(f"Submission file created successfully at {C.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head of submission:")
        print(df_sub.head())

        expected_cols = ["contact_id", "contact"]
        assert all(
            col in df_sub.columns for col in expected_cols
        ), f"Missing columns in submission. Expected {expected_cols}"
        assert (
            df_sub["contact"].isin([0, 1]).all()
        ), "Contact column contains non-binary values"
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {C.SUBMISSION_PATH}"
        )

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    main()
