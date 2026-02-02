import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library components
# We import modules to patch their global variables later
import library.config as lib_config
import library.utils as lib_utils
import library.model as lib_model
import library.data_loader as lib_loader
import library.train as lib_train


def main():
    # 1. Setup
    print("=== Setting up Demo Environment ===")
    lib_utils.seed_everything(42)

    # Define demo directories
    DEMO_DIR = "./working/demo_task"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    DEMO_DATA_DIR = os.path.join(DEMO_DIR, "data")
    os.makedirs(DEMO_DATA_DIR)

    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(DEMO_SUBMISSION_DIR)

    # 2. Create Synthetic Data (Mocking)
    # We read the actual schema to ensure feature engineering works
    print("=== Generating Synthetic Data for Speed ===")
    real_train_path = "./metadata/train.parquet"
    if os.path.exists(real_train_path):
        # Read only schema/head
        df_schema = pd.read_parquet(real_train_path).head(1)
        columns = df_schema.columns.tolist()
    else:
        # Fallback if metadata missing (unlikely based on prompt)
        raise FileNotFoundError(f"Metadata not found at {real_train_path}")

    # Generate small datasets (100 rows each)
    num_rows = 100

    # Create random data preserving types
    mock_data = {}
    for col in columns:
        if "Id" in col or "Cover_Type" in col:
            mock_data[col] = np.random.randint(1, 8, num_rows)  # Classes 1-7
        elif col.startswith("Soil_Type") or col.startswith("Wilderness_Area"):
            mock_data[col] = np.random.randint(0, 2, num_rows)
        else:
            mock_data[col] = np.random.randn(num_rows) * 100

    df_mock = pd.DataFrame(mock_data)

    # Ensure IDs are unique floats as per schema usually
    df_mock["Id"] = np.arange(num_rows, dtype=float)

    # Save Train
    demo_train_path = os.path.join(DEMO_DATA_DIR, "train.parquet")
    df_mock.to_parquet(demo_train_path)

    # Save Val (same structure)
    demo_val_path = os.path.join(DEMO_DATA_DIR, "val.parquet")
    df_mock.to_parquet(demo_val_path)

    # Save Test (drop target)
    demo_test_path = os.path.join(DEMO_DATA_DIR, "test.parquet")
    df_mock.drop(columns=["Cover_Type"]).to_parquet(demo_test_path)

    print(f"Created mock datasets at {DEMO_DATA_DIR}")

    # 3. Patch Library Paths
    # We redirect the library modules to use our demo data and demo output folders
    # This allows us to use the library functions as-is without modifying the files

    print("=== Patching Library Configuration ===")

    # Define new cache paths
    demo_train_cache = os.path.join(DEMO_DIR, "train_processed.npy")
    demo_train_labels = os.path.join(DEMO_DIR, "train_labels.npy")
    demo_val_cache = os.path.join(DEMO_DIR, "val_processed.npy")
    demo_val_labels = os.path.join(DEMO_DIR, "val_labels.npy")
    demo_test_cache = os.path.join(DEMO_DIR, "test_processed.npy")
    demo_test_ids = os.path.join(DEMO_DIR, "test_ids.npy")

    demo_model_path = os.path.join(DEMO_DIR, "demo_model.pth")
    demo_sub_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Patch data_loader module
    lib_loader.TRAIN_DATA_PATH = demo_train_path
    lib_loader.VAL_DATA_PATH = demo_val_path
    lib_loader.TEST_DATA_PATH = demo_test_path
    lib_loader.WORKING_DIR = DEMO_DIR
    lib_loader.TRAIN_CACHE_PATH = demo_train_cache
    lib_loader.TRAIN_LABELS_PATH = demo_train_labels
    lib_loader.VAL_CACHE_PATH = demo_val_cache
    lib_loader.VAL_LABELS_PATH = demo_val_labels
    lib_loader.TEST_CACHE_PATH = demo_test_cache
    lib_loader.TEST_IDS_PATH = demo_test_ids

    # Patch train module
    lib_train.MODEL_SAVE_PATH = demo_model_path
    lib_train.SUBMISSION_PATH = demo_sub_path

    # Patch model module (just in case it's used directly)
    lib_model.TRAIN_DATA_PATH = demo_train_path
    # ... (other paths in model.py are less critical if we use train.py but good practice)

    # 4. Verify Components
    print("=== Verifying Model Components ===")

    # Test Model Architecture
    input_dim = 54 + 4  # 54 original + 4 engineered features
    model = lib_model.ParallelDCNSEResNet(
        input_dim=input_dim,
        num_classes=7,
        hidden_dim=64,
        num_cross_layers=2,
        se_reduction=4,
        dropout=0.1,
    )

    # Forward pass check
    dummy_input = torch.randn(10, input_dim)
    output = model(dummy_input)

    assert output.shape == (
        10,
        7,
    ), f"Model output shape mismatch. Expected (10, 7), got {output.shape}"
    print("Model forward pass verification successful.")

    # Test Utils
    print("=== Verifying Utils ===")
    checkpoint = lib_utils.ModelCheckpoint(mode="max")
    assert checkpoint.best_score == float("-inf")

    # Simulate improvement
    improved = checkpoint.step(0.5, model)
    assert improved is True
    assert checkpoint.best_score == 0.5
    assert checkpoint.best_state is not None

    # Simulate no improvement
    improved = checkpoint.step(0.4, model)
    assert improved is False
    assert checkpoint.best_score == 0.5
    print("ModelCheckpoint verification successful.")

    # 5. Run Training Pipeline
    print("=== Running Training Pipeline (Integration Test) ===")

    # We use very small epochs and batch size for speed
    # load_cached_data=False forces the loader to read our new patched parquet files
    try:
        lib_train.run_training(
            epochs=2, batch_size=16, learning_rate=1e-3, load_cached_data=False
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # 6. Verify Outputs
    print("=== Verifying Outputs ===")

    if not os.path.exists(demo_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {demo_model_path}")

    if not os.path.exists(demo_sub_path):
        raise FileNotFoundError(f"Submission file not found at {demo_sub_path}")

    # Check submission content
    df_sub = pd.read_csv(demo_sub_path)
    assert "Id" in df_sub.columns
    assert "Cover_Type" in df_sub.columns
    assert len(df_sub) == num_rows

    print(f"Success! Submission generated with {len(df_sub)} rows.")
    print("All demonstrations and verifications passed.")


if __name__ == "__main__":
    main()
