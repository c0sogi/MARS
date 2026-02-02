import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, CheckpointManager, calculate_robust_auc
from library.data import prepare_folds, get_dataloaders, get_test_dataloaders
from library.models import BirdCNN, BirdMLP
from library.engine import train_one_epoch, validate


def run_demo():
    print("Starting Demo Execution...")

    # 1. Setup
    # Initialize directories
    Config.setup()
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Device configuration
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n--- Step 1: Preparing Data Folds ---")
    # This will create folds.parquet in the cache directory
    # We force regeneration to ensure the demo tests the logic
    df_folds = prepare_folds(load_cached_data=False)

    assert os.path.exists(
        os.path.join(Config.CACHE_DIR, "folds.parquet")
    ), "Folds file was not cached!"
    assert "fold" in df_folds.columns, "Fold column missing in dataframe"
    print(f"Folds prepared. Total samples: {len(df_folds)}")

    # 3. Data Loading
    print("\n--- Step 2: Testing Data Loaders (Fold 0) ---")
    # We use a small batch size for the demo
    demo_batch_size = 8
    loaders = get_dataloaders(
        fold=0, df_folds=df_folds, batch_size=demo_batch_size, num_workers=0
    )

    # Test CNN Loader
    cnn_loader = loaders["cnn"]["train"]
    # Fetch one batch
    cnn_batch = next(iter(cnn_loader))
    cnn_inputs, cnn_targets = cnn_batch

    print(f"CNN Batch Input Shape: {cnn_inputs.shape}")
    print(f"CNN Batch Target Shape: {cnn_targets.shape}")

    # Assertions for CNN
    # Shape: (B, 3, 224, 224)
    assert cnn_inputs.shape == (
        demo_batch_size,
        3,
        224,
        224,
    ), f"Unexpected CNN input shape: {cnn_inputs.shape}"
    # Target: (B, 19)
    assert cnn_targets.shape == (
        demo_batch_size,
        Config.NUM_CLASSES,
    ), f"Unexpected CNN target shape: {cnn_targets.shape}"

    # Test MLP Loader
    mlp_loader = loaders["mlp"]["train"]
    mlp_batch = next(iter(mlp_loader))
    mlp_inputs, mlp_targets = mlp_batch

    print(f"MLP Batch Input Shape: {mlp_inputs.shape}")

    # Assertions for MLP
    # Shape: (B, 100)
    assert mlp_inputs.shape == (
        demo_batch_size,
        Config.MLP_INPUT_DIM,
    ), f"Unexpected MLP input shape: {mlp_inputs.shape}"
    assert mlp_targets.shape == (
        demo_batch_size,
        Config.NUM_CLASSES,
    ), f"Unexpected MLP target shape: {mlp_targets.shape}"

    # 4. Model Initialization & Forward Pass
    print("\n--- Step 3: Model Initialization & Forward Pass ---")

    # CNN Model (ResNet18) - Using pretrained=False for speed/offline safety in demo
    # In a real run, pretrained=True is recommended
    cnn_model = BirdCNN(model_name="resnet18", pretrained=False).to(device)
    cnn_output = cnn_model(cnn_inputs.to(device))

    print(f"CNN Output Shape: {cnn_output.shape}")
    assert cnn_output.shape == (
        demo_batch_size,
        Config.NUM_CLASSES,
    ), "CNN output shape mismatch"

    # MLP Model
    mlp_model = BirdMLP().to(device)
    mlp_output = mlp_model(mlp_inputs.to(device))

    print(f"MLP Output Shape: {mlp_output.shape}")
    assert mlp_output.shape == (
        demo_batch_size,
        Config.NUM_CLASSES,
    ), "MLP output shape mismatch"

    # 5. Training Loop Demonstration
    print("\n--- Step 4: Training Loop Demo (1 Epoch) ---")

    # Setup Optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
    scheduler = None  # Skip scheduler for simple demo

    # Train CNN for 1 epoch
    print("Training CNN...")
    train_loss = train_one_epoch(
        model=cnn_model,
        dataloader=cnn_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=1,
    )
    print(f"CNN Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate CNN
    print("Validating CNN...")
    val_loader = loaders["cnn"]["val"]
    val_loss, val_auc, val_preds, val_targets = validate(cnn_model, val_loader, device)

    print(f"CNN Val Loss: {val_loss:.4f}")
    print(f"CNN Val AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    # AUC might be 0.0 if the random init model predicts poorly or classes are missing in batch,
    # but robust_auc handles missing classes. Range check:
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"
    assert (
        val_preds.shape == val_targets.shape
    ), "Prediction and target shapes mismatch in validation"

    # 6. Checkpoint Management
    print("\n--- Step 5: Checkpoint Management ---")
    ckpt_manager = CheckpointManager(
        model_name="resnet18_demo", fold=0, save_dir=Config.CHECKPOINT_DIR
    )
    saved = ckpt_manager.save(cnn_model, val_auc, epoch=1)

    print(f"Checkpoint saved: {saved}")
    expected_path = os.path.join(
        Config.CHECKPOINT_DIR, f"resnet18_demo_fold_0_epoch_1_auc_{val_auc:.5f}.pth"
    )
    assert os.path.exists(
        expected_path
    ), f"Checkpoint file not found at {expected_path}"

    # 7. Inference Demonstration
    print("\n--- Step 6: Inference Demo ---")
    test_loaders = get_test_dataloaders(batch_size=demo_batch_size, num_workers=0)
    test_cnn_loader = test_loaders["cnn"]
    test_ids = test_loaders["ids"]

    print(f"Number of test samples: {len(test_ids)}")

    # Run inference on one batch
    cnn_model.eval()
    with torch.no_grad():
        test_inputs, test_rec_ids = next(iter(test_cnn_loader))
        test_outputs = cnn_model(test_inputs.to(device))
        test_probs = torch.sigmoid(test_outputs)

    print(f"Test Batch Probabilities Shape: {test_probs.shape}")
    assert test_probs.shape[1] == Config.NUM_CLASSES, "Test output classes mismatch"
    assert (
        test_probs.min() >= 0 and test_probs.max() <= 1
    ), "Probabilities out of [0, 1] range"

    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
