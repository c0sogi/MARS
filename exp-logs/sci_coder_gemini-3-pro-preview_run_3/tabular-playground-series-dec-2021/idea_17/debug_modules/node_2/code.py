import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import LowRankCrossLayer, ResNetBlock, ParallelDCNResNet
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True  # Use subset of data (10,000 rows)
    Config.DEBUG_SAMPLE_SIZE = 5000  # Even smaller for this demo script to be very fast
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64

    # Set up a specific directory for this demo run
    demo_dir = "./working/demo_execution"
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission", "submission.csv")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Seeding
    seed_everything(42)
    val1 = np.random.rand()
    seed_everything(42)
    val2 = np.random.rand()
    assert val1 == val2, "seed_everything failed to produce deterministic numpy results"
    print("seed_everything: Verified.")

    # Test Device
    device = get_device()
    print(f"Device: {device}")
    assert isinstance(
        device, torch.device
    ), "get_device did not return a torch.device object"

    # -------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading & Feature Engineering...")

    # Force reprocessing to test the pipeline (load_cached_data=False)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verify Train Loader
    print(f"Train Loader Batches: {len(train_loader)}")
    X_batch, y_batch = next(iter(train_loader))

    print(f"Batch X Shape: {X_batch.shape}")
    print(f"Batch y Shape: {y_batch.shape}")

    # Assertions
    assert X_batch.dim() == 2, "Input batch should be 2D (Batch, Features)"
    assert y_batch.dim() == 1, "Target batch should be 1D (Batch)"
    assert (
        X_batch.shape[0] == Config.BATCH_SIZE
        or X_batch.shape[0] <= Config.DEBUG_SAMPLE_SIZE
    )

    input_dim = X_batch.shape[1]
    print(f"Detected Input Dimension: {input_dim}")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Validation
    # -------------------------------------------------------------------------
    print("\n[4] Validating Model Components...")

    # Test LowRankCrossLayer
    dcn_rank = 8
    layer = LowRankCrossLayer(input_dim=input_dim, rank=dcn_rank).to(device)
    dummy_input = torch.randn(32, input_dim).to(device)

    # DCN Layer requires x0 (initial) and xl (current)
    out = layer(dummy_input, dummy_input)
    assert (
        out.shape == dummy_input.shape
    ), f"DCN Layer output shape mismatch. Expected {dummy_input.shape}, got {out.shape}"
    print("LowRankCrossLayer: Forward pass successful.")

    # Test ResNetBlock
    hidden_dim = 64
    res_block = ResNetBlock(hidden_dim=hidden_dim, dropout_rate=0.1).to(device)
    dummy_hidden = torch.randn(32, hidden_dim).to(device)
    out_res = res_block(dummy_hidden)
    assert out_res.shape == dummy_hidden.shape, "ResNetBlock output shape mismatch"
    print("ResNetBlock: Forward pass successful.")

    # Test Full Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_rank=Config.DCN_RANK,
        resnet_hidden=Config.RESNET_HIDDEN_DIM,
        resnet_blocks=Config.RESNET_NUM_BLOCKS,
    ).to(device)

    logits = model(dummy_input)
    assert logits.shape == (
        32,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (32, {Config.NUM_CLASSES}), got {logits.shape}"
    print("ParallelDCNResNet: Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Pipeline (1 Epoch)...")

    # We use load_cached_data=True now because we generated cache in step 3
    trained_model, _, _ = run_training(
        epochs=1, learning_rate=1e-3, load_cached_data=True, debug=True
    )

    # Verify Model File Exists
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file was not saved at {Config.MODEL_PATH}"
    print(f"Model successfully saved to {Config.MODEL_PATH}")

    # Verify returned object is a model
    assert isinstance(trained_model, torch.nn.Module)

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference Pipeline...")

    run_inference(
        load_cached_data=True,
        model_path=Config.MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
        debug=True,
    )

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    # Verify Columns
    expected_cols = ["Id", "Cover_Type"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Verify Content
    assert not df_sub.empty, "Submission file is empty"
    assert df_sub["Id"].dtype == "int64" or df_sub["Id"].dtype == "int32"
    assert (
        df_sub["Cover_Type"].dtype == "int64" or df_sub["Cover_Type"].dtype == "int32"
    )

    print("Inference Pipeline: Verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
