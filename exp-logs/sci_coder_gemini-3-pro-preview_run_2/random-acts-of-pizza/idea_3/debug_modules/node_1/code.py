import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import load_data
from library.feature_engineering import process_data, DualBranchPreprocessor
from library.model_factory import (
    get_linear_base_model,
    get_tree_base_model,
    get_meta_model,
)
from library.stacking_engine import StackingEngine


def main():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> Setting up configuration for fast demonstration...")

    # Set fixed seed
    set_seed(42)

    # Override Config parameters to ensure the script runs quickly
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Must be > N_PCA_COMPONENTS (32)
    Config.N_FOLDS = 2  # Minimal folds for CV

    # Use a separate working directory for this demo to avoid messing with real cache
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Reduce model complexity for speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["min_child_samples"] = (
        5  # Reduce constraint for small sample size
    )
    Config.LR_PARAMS["max_iter"] = 10

    # Setup directories
    Config.setup()

    print(f"DEBUG Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n>>> Testing Data Loader...")
    # Force reload to ignore any existing cache in the default directory,
    # and to ensure we use the DEBUG sample size.
    df_train, df_val, df_test = load_data(load_cached_data=False)

    # Assertions for Data Loader
    assert isinstance(df_train, pd.DataFrame), "df_train should be a DataFrame"
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} training samples, got {len(df_train)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} test samples, got {len(df_test)}"
    assert "request_text_edit_aware" in df_train.columns, "Text column missing"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train"

    print("Data Loader assertions passed.")

    # 3. Feature Engineering
    print("\n>>> Testing Feature Engineering...")

    # We use process_data which handles the DualBranchPreprocessor internally
    # load_cached_data=False ensures we process the small debug dataframes we just loaded
    data = process_data(df_train, df_val, df_test, load_cached_data=False)

    # Assertions for Feature Engineering
    expected_keys = [
        "train_linear",
        "train_tree",
        "y_train",
        "val_linear",
        "val_tree",
        "y_val",
        "test_linear",
        "test_tree",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in processed data: {key}"
        assert isinstance(data[key], np.ndarray), f"{key} should be a numpy array"

    # Check dimensions
    # Linear branch: 384 (embedding) + 10 (numerical) = 394 features
    # Tree branch: 32 (PCA) + 10 (numerical) = 42 features
    n_numerical = len(Config.NUMERICAL_COLS)
    n_pca = Config.N_PCA_COMPONENTS

    # Note: SentenceTransformer 'all-MiniLM-L6-v2' outputs 384 dimensions
    expected_linear_dim = 384 + n_numerical
    expected_tree_dim = n_pca + n_numerical

    assert (
        data["train_linear"].shape[1] == expected_linear_dim
    ), f"Expected {expected_linear_dim} linear features, got {data['train_linear'].shape[1]}"
    assert (
        data["train_tree"].shape[1] == expected_tree_dim
    ), f"Expected {expected_tree_dim} tree features, got {data['train_tree'].shape[1]}"

    print("Feature Engineering assertions passed.")

    # 4. Model Factory Verification
    print("\n>>> Testing Model Factory...")
    lr_model = get_linear_base_model()
    tree_model = get_tree_base_model()
    meta_model = get_meta_model()

    assert hasattr(lr_model, "fit"), "Linear model missing fit method"
    assert hasattr(tree_model, "fit"), "Tree model missing fit method"
    assert hasattr(meta_model, "fit"), "Meta model missing fit method"

    print("Model Factory assertions passed.")

    # 5. Stacking Engine Execution
    print("\n>>> Testing Stacking Engine (Training & Inference)...")
    engine = StackingEngine()

    # Run the full pipeline: CV -> Meta Training -> Final Retraining -> Prediction
    predictions = engine.run(data)

    # Assertions for Stacking Engine
    assert len(predictions) == len(
        df_test
    ), f"Prediction count {len(predictions)} does not match test set size {len(df_test)}"

    # Check probabilities range
    assert np.all(predictions >= 0.0) and np.all(
        predictions <= 1.0
    ), "Predictions must be probabilities between 0 and 1"

    print(f"Generated {len(predictions)} predictions.")
    print("Stacking Engine assertions passed.")

    # 6. Submission Generation
    print("\n>>> Testing Submission Generation...")
    output_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    save_submission(
        request_ids=df_test["request_id"].values,
        predictions=predictions,
        output_path=output_path,
    )

    assert os.path.exists(output_path), "Submission file was not created"

    # Verify content format
    sub_df = pd.read_csv(output_path)
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns are incorrect"
    assert len(sub_df) == len(df_test), "Submission row count mismatch"

    print("Submission generation assertions passed.")

    print("\n>>> All demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
