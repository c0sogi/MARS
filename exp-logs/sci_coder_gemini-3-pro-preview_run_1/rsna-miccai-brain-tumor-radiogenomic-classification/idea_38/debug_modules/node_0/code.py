import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.engine as engine


def run_pipeline_demo():
    print("=" * 40)
    print(" MGMT Promoter Methylation Prediction Demo")
    print("=" * 40)

    # ---------------------------------------------------------
    # 1. Configuration & Patching for Speed/Safety
    # ---------------------------------------------------------
    print("\n[Setup] Configuring environment for rapid demonstration...")

    # Define a temporary working directory for this demo
    demo_dir = os.path.join(config.WORKING_DIR, "demo_execution_custom")
    os.makedirs(demo_dir, exist_ok=True)

    # Define paths for demo artifacts
    demo_train_cache = os.path.join(demo_dir, "train_cache.npy")
    demo_val_cache = os.path.join(demo_dir, "val_cache.npy")
    demo_test_cache = os.path.join(demo_dir, "test_cache.npy")
    demo_model_path = os.path.join(demo_dir, "model.pth")
    demo_sub_path = os.path.join(demo_dir, "submission.csv")

    # Patch 'library.dataset' globals to force debug mode and use demo cache paths
    # This ensures we only process a tiny subset of data (6 samples)
    dataset.DEBUG = True
    dataset.DEBUG_DATA_SIZE = 6
    dataset.TRAIN_CACHE_FILE = demo_train_cache
    dataset.VAL_CACHE_FILE = demo_val_cache
    dataset.TEST_CACHE_FILE = demo_test_cache

    # Patch 'library.engine' globals to redirect model saving
    engine.MODEL_SAVE_PATH = demo_model_path

    # Patch 'library.config' globals (used by other modules)
    config.DEBUG = True
    config.BATCH_SIZE = 2  # Small batch size for demo
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    print(f"Debug Mode: {dataset.DEBUG}")
    print(f"Debug Data Size: {dataset.DEBUG_DATA_SIZE}")
    print(f"Working Directory: {demo_dir}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n[Step 1] Loading and Processing Data...")

    # load_cached_data=False forces the pipeline to read DICOMs and process them
    train_ds, val_ds, test_ds = dataset.get_datasets(load_cached_data=False)

    print(f"Train Set Size: {len(train_ds)}")
    print(f"Val Set Size:   {len(val_ds)}")
    print(f"Test Set Size:  {len(test_ds)}")

    # Verify we respected the debug limit
    assert (
        len(train_ds) <= dataset.DEBUG_DATA_SIZE
    ), "Train dataset exceeded debug size limit."

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[Step 2] Building VAANet Model...")
    device = utils.get_device()
    print(f"Compute Device: {device}")

    net = model_lib.build_model()
    net = net.to(device)

    # Sanity check: Pass a dummy tensor to verify architecture
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[Step 3] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run training for 1 epoch
    best_score = engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=1,  # Override default to 1 for speed
        patience=1,
    )

    print(f"Training finished. Best Validation AUC: {best_score:.4f}")
    assert os.path.exists(demo_model_path), "Model checkpoint was not saved."

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[Step 4] Generating Submission...")

    engine.generate_submission(
        model=net, test_loader=test_loader, device=device, output_path=demo_sub_path
    )

    # ---------------------------------------------------------
    # 6. Verification
    # ---------------------------------------------------------
    print("\n[Step 5] Validating Output...")

    assert os.path.exists(demo_sub_path), "Submission file missing."

    df_sub = pd.read_csv(demo_sub_path)
    print("Submission Head:")
    print(df_sub.head())

    # Check Columns
    assert "BraTS21ID" in df_sub.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Missing MGMT_value column"

    # Check Row Count
    assert len(df_sub) == len(
        test_ds
    ), f"Submission row count {len(df_sub)} mismatch with test set {len(test_ds)}"

    # Check Probability Range
    probs = df_sub["MGMT_value"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"

    print("\n" + "=" * 40)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    # Ensure reproducibility
    config.seed_everything(config.SEED)
    run_pipeline_demo()
