import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config, DataConfig, TrainConfig, ModelConfig
from library.utils import seed_everything, get_device
from library.data import feature_engineering, CoverTypeDataset, get_dataloaders
from library.model import ParallelLowRankDCNResNet, LowRankCrossLayer, ResNetBlock
from library.train import train_one_epoch, validate, run_training
from library.inference import predict


def generate_dummy_data(num_samples, is_test=False):
    """
    Generates a dummy DataFrame matching the schema expected by DataConfig.
    """
    data = {}

    # 1. ID Column
    data[DataConfig.ID_COL] = np.arange(num_samples)

    # 2. Target Column (only for train/val)
    if not is_test:
        # Classes are 1-7
        data[DataConfig.TARGET_COL] = np.random.randint(1, 8, size=num_samples)

    # 3. Raw Continuous Features
    for col in DataConfig.RAW_CONT_COLS:
        data[col] = np.random.randn(num_samples) * 100 + 2000  # Random plausible values

    # 4. Binary Features (Wilderness Areas and Soil Types)
    for col in DataConfig.BINARY_COLS:
        data[col] = np.random.randint(0, 2, size=num_samples)

    return pd.DataFrame(data)


def main():
    print("Starting demonstration script...")

    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    seed_everything(42)

    # Define temporary directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    cache_dir = os.path.join(demo_dir, "cache")
    submission_dir = os.path.join(demo_dir, "submission")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print(f"Created temporary working directory: {demo_dir}")

    # Override Config paths to use our dummy environment
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = cache_dir
    Config.SUBMISSION_DIR = submission_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Override TrainConfig for speed
    TrainConfig.EPOCHS = 2
    TrainConfig.BATCH_SIZE = 8
    TrainConfig.VAL_CHECK_INTERVAL = 1

    # 2. Data Generation & Metadata Setup
    # ---------------------------------------------------------
    print("\n[1/5] Generating dummy data...")

    # Create small datasets
    df_train = generate_dummy_data(100, is_test=False)
    df_val = generate_dummy_data(50, is_test=False)
    df_test = generate_dummy_data(50, is_test=True)

    # Save as parquet (simulating the metadata files)
    train_path = os.path.join(demo_dir, "train.parquet")
    val_path = os.path.join(demo_dir, "val.parquet")
    test_path = os.path.join(demo_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    # Point Config to these files
    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path

    print("Dummy data generated and saved.")

    # 3. Verify Data Pipeline
    # ---------------------------------------------------------
    print("\n[2/5] Verifying Data Pipeline...")

    # Test Feature Engineering
    print("  Testing feature_engineering()...")
    df_eng = feature_engineering(df_train.copy())

    # Check if new columns were added
    expected_new_cols = DataConfig.NEW_CONT_COLS
    for col in expected_new_cols:
        assert col in df_eng.columns, f"Feature engineering failed to create {col}"
    print("  Feature engineering verification passed.")

    # Test Data Loading
    print("  Testing get_dataloaders()...")
    # Force load_cached_data=False to ensure it processes our new dummy files
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple script execution
        load_cached_data=False,
    )

    # Verify Batch Shapes
    xb, yb = next(iter(train_loader))
    assert xb.shape[0] == TrainConfig.BATCH_SIZE, "Incorrect batch size"
    assert (
        xb.shape[1] == DataConfig.INPUT_DIM
    ), f"Incorrect input dim: {xb.shape[1]} vs {DataConfig.INPUT_DIM}"
    assert yb.shape[0] == TrainConfig.BATCH_SIZE, "Incorrect target batch size"
    print(f"  DataLoader shapes verified: Input {xb.shape}, Target {yb.shape}")

    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture...")

    device = get_device()

    # Test LowRankCrossLayer
    print("  Testing LowRankCrossLayer...")
    dcn_layer = LowRankCrossLayer(in_features=DataConfig.INPUT_DIM, rank=4).to(device)
    dummy_input = torch.randn(TrainConfig.BATCH_SIZE, DataConfig.INPUT_DIM).to(device)
    out_dcn = dcn_layer(dummy_input, dummy_input)
    assert out_dcn.shape == dummy_input.shape, "LowRankCrossLayer output shape mismatch"

    # Test ResNetBlock
    print("  Testing ResNetBlock...")
    res_block = ResNetBlock(hidden_dim=32).to(device)
    dummy_hidden = torch.randn(TrainConfig.BATCH_SIZE, 32).to(device)
    out_res = res_block(dummy_hidden)
    assert out_res.shape == dummy_hidden.shape, "ResNetBlock output shape mismatch"

    # Test Full Model
    print("  Testing ParallelLowRankDCNResNet...")
    model = ParallelLowRankDCNResNet().to(device)
    logits = model(dummy_input)
    assert logits.shape == (
        TrainConfig.BATCH_SIZE,
        DataConfig.NUM_CLASSES,
    ), f"Model output shape mismatch: {logits.shape}"
    print("  Model architecture verification passed.")

    # 5. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[4/5] Verifying Training Loop...")

    # We use run_training which handles the loop, validation, and saving
    trained_model = run_training(train_loader, val_loader)

    # Check if model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"
    print(f"  Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 6. Verify Inference
    # ---------------------------------------------------------
    print("\n[5/5] Verifying Inference...")

    predict(
        model_path=Config.MODEL_SAVE_PATH,
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=0,
        output_path=Config.SUBMISSION_PATH,
    )

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_sub.shape}")

    # Verify submission content
    assert list(df_sub.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"
    assert len(df_sub) == len(
        df_test
    ), f"Submission length mismatch: {len(df_sub)} vs {len(df_test)}"
    assert df_sub["Cover_Type"].min() >= 1, "Invalid class label (min < 1)"
    assert df_sub["Cover_Type"].max() <= 7, "Invalid class label (max > 7)"

    print("  Inference verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
