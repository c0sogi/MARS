import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import provided library modules
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data_loader as data_loader
import library.train as train_lib


def run_demo():
    print("===========================================================")
    print("   Iceberg Classification Pipeline Demonstration")
    print("===========================================================")

    # 1. Setup and Configuration
    print("\n[1] Environment Setup")
    utils.seed_everything(config.SEED)
    print(f"    Random Seed: {config.SEED}")
    print(f"    Device: {config.DEVICE}")
    print(f"    Working Directory: {config.WORKING_DIR}")

    # 2. Data Processing and Loading
    print("\n[2] Data Processing & Loading")
    # process_and_cache_data handles loading JSONs, normalizing, and caching
    # We load the data to verify shapes and integrity
    X_train, y_train, inc_train, X_test, inc_test, ids_test = (
        data_loader.process_and_cache_data()
    )

    print(f"    Training Data Shape: {X_train.shape}")
    print(f"    Test Data Shape: {X_test.shape}")

    # Assertions to verify data integrity
    assert (
        len(X_train) == len(y_train) == len(inc_train)
    ), "Training data dimension mismatch"
    assert X_train.shape[1] == 3, "Expected 3 channels (Band 1, Band 2, Mean)"
    assert X_train.shape[2:] == (75, 75), "Expected 75x75 image size"
    assert not np.isnan(X_train).any(), "Training data contains NaNs"

    # 3. DataLoader Verification
    print("\n[3] DataLoader Verification (Fold 0)")
    # Get loaders for the first fold
    train_loader, val_loader = data_loader.get_dataloaders(fold_idx=0)

    # Fetch a single batch to verify structure
    imgs, angs, lbls = next(iter(train_loader))
    print(f"    Batch Images Shape: {imgs.shape}")
    print(f"    Batch Angles Shape: {angs.shape}")
    print(f"    Batch Labels Shape: {lbls.shape}")

    # Assertions for tensor shapes
    assert imgs.shape == (config.BATCH_SIZE, 3, 75, 75)
    assert angs.shape == (config.BATCH_SIZE, 1)
    assert lbls.shape == (config.BATCH_SIZE, 1)
    assert imgs.dtype == torch.float32

    # 4. Model Initialization
    print("\n[4] Model Initialization (GA_WBN)")
    model = model_lib.GA_WBN().to(config.DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model Parameters: {num_params:,}")

    # Verify Forward Pass
    imgs, angs = imgs.to(config.DEVICE), angs.to(config.DEVICE)
    with torch.no_grad():
        output = model(imgs, angs)

    print(f"    Forward Pass Output Shape: {output.shape}")
    assert output.shape == (config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # 5. Training Loop Demonstration
    print("\n[5] Training Loop Demonstration (1 Epoch)")
    # We run a manual loop for 1 epoch to demonstrate usage without waiting for full training
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train for 1 epoch
    train_loss = train_lib.train_epoch(
        model, train_loader, optimizer, criterion, config.DEVICE
    )
    print(f"    Epoch 1 Train Loss: {train_loss:.6f}")

    # Validate
    val_loss = train_lib.validate(model, val_loader, criterion, config.DEVICE)
    print(f"    Epoch 1 Val Loss:   {val_loss:.6f}")

    assert train_loss > 0, "Train loss should be positive"
    assert val_loss > 0, "Validation loss should be positive"

    # 6. Checkpointing
    print("\n[6] Checkpointing Verification")
    demo_ckpt_path = os.path.join(config.WORKING_DIR, "demo_model.pth")
    utils.save_checkpoint(model, demo_ckpt_path)
    print(f"    Model saved to: {demo_ckpt_path}")

    assert os.path.exists(demo_ckpt_path), "Checkpoint file was not created"

    # Load model back to verify
    model_loaded = model_lib.GA_WBN().to(config.DEVICE)
    model_loaded = utils.load_checkpoint(model_loaded, demo_ckpt_path, config.DEVICE)

    # Check if weights match
    p_orig = next(model.parameters())
    p_load = next(model_loaded.parameters())
    assert torch.equal(p_orig, p_load), "Loaded model weights do not match original"
    print("    Model loaded successfully and weights match.")

    # 7. Inference Demonstration
    print("\n[7] Inference Demonstration")
    test_loader, test_ids = data_loader.get_test_loader()

    model.eval()
    predictions = []

    # Run inference on just the first batch to save time
    with torch.no_grad():
        imgs_test, angs_test = next(iter(test_loader))
        imgs_test, angs_test = imgs_test.to(config.DEVICE), angs_test.to(config.DEVICE)

        logits = model(imgs_test, angs_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        predictions.append(probs)

    print(f"    Inference Batch Shape: {probs.shape}")
    print(f"    Sample Probability: {probs[0][0]:.4f}")

    assert probs.shape == (len(imgs_test), 1), "Prediction shape mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print("\n===========================================================")
    print("   Demonstration Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    run_demo()
