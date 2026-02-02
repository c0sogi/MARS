import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shutil
import library.config as config
import library.utils as utils
import library.features as features
import library.data as data
import library.model as model


# ==========================================
# 1. Setup & Configuration
# ==========================================
def run_demo():
    print("Initializing Demo Execution...")

    # Set seeds
    np.random.seed(config.RANDOM_SEED)

    # Define working paths for demo
    DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create a small subset of test data for speed
    print("Creating subset of test metadata for rapid inference...")
    full_test = pd.read_csv(config.TEST_METADATA_PATH)
    small_test = full_test.sample(n=100, random_state=config.RANDOM_SEED).sort_values(
        "id"
    )
    small_test_path = os.path.join(DEMO_DIR, "test_subset.csv")
    small_test.to_csv(small_test_path, index=False)

    # ==========================================
    # 2. Patching Library for Speed
    # ==========================================
    print("Patching library configurations for fast execution...")

    # Patch XGBoost parameters in library.model
    # We use very few estimators and shallow trees for the demo
    demo_xgb_params = model.XGB_PARAMS.copy()
    demo_xgb_params.update(
        {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_jobs": 4,
            "tree_method": "hist",
        }
    )
    model.XGB_PARAMS = demo_xgb_params

    # Patch Test Metadata Path in library.data
    # This ensures get_test_data reads our small file
    data.TEST_METADATA_PATH = small_test_path

    # Patch Model Save Directory to keep things organized
    demo_model_dir = os.path.join(DEMO_DIR, "xgb_models")
    model.MODEL_SAVE_DIR = demo_model_dir
    os.makedirs(demo_model_dir, exist_ok=True)

    # Patch Submission Path
    demo_sub_path = os.path.join(DEMO_DIR, "submission.csv")
    model.SUBMISSION_FILE_PATH = demo_sub_path

    # ==========================================
    # 3. Data Pipeline Execution
    # ==========================================
    print("\nStep 1: Data Management")
    dm = data.DataManager()

    # Get Training Data (Debug Mode)
    # This will process structures (full) but only return sampled training pairs
    print("Generating training data (Debug Mode)...")
    X_train, y_train, X_val, y_val = dm.get_train_data(
        load_cached_data=True, debug_mode=True
    )

    # Validation of Data Shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    assert not X_train.empty, "Training data should not be empty"
    assert len(X_train) == len(y_train), "Mismatch in training features and targets"
    assert (
        "type" in X_train.columns
    ), "Type column missing from X_train (needed for stratification)"

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("\nStep 2: Model Training")
    strat_model = model.StratifiedModel()

    # Update the internal directory of the instance since we patched the class variable
    strat_model.model_dir = demo_model_dir

    strat_model.train(X_train, y_train, X_val, y_val)

    # Verify models were saved
    saved_models = os.listdir(demo_model_dir)
    print(f"Saved models: {saved_models}")
    assert len(saved_models) > 0, "No models were saved during training"

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\nStep 3: Inference")

    # Get Test Data (using patched path)
    X_test, test_ids = dm.get_test_data(load_cached_data=True)

    print(f"X_test shape: {X_test.shape}")
    assert len(X_test) == 100, f"Expected 100 test samples, got {len(X_test)}"

    # Predict
    submission = strat_model.predict(X_test, test_ids)

    # ==========================================
    # 6. Final Validation
    # ==========================================
    print("\nStep 4: Validation")

    # Check Submission File
    assert os.path.exists(demo_sub_path), "Submission file was not created"

    # Check Content
    df_sub = pd.read_csv(demo_sub_path)
    print("Submission Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "id",
        "scalar_coupling_constant",
    ], "Incorrect submission columns"
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        not df_sub["scalar_coupling_constant"].isnull().any()
    ), "NaN values found in predictions"

    # Check if predictions are somewhat reasonable (not all zero, unless data dictates it)
    # With 10 estimators, accuracy is low, but variance should exist
    if len(df_sub) > 1:
        std_pred = df_sub["scalar_coupling_constant"].std()
        print(f"Prediction Std Dev: {std_pred}")

    print("\nDemo Execution Completed Successfully!")


if __name__ == "__main__":
    run_demo()
