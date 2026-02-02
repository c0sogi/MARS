import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Ensure library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_train_val_loaders, get_test_loader
from library.models import CactusRepVGG, CactusResNet, CactusNeXt
from library.engine import train_one_epoch, validate, SWAHandler, inference_tta


def main():
    print("=== Starting Cactus Identification Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment...")

    # Override Config for a quick demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 64  # Use a small subset for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data...")

    # We force load_cached_data=False to demonstrate the raw processing logic.
    # This reads from metadata CSVs and loads images from disk.
    train_loader, val_loader, fs_stats = get_train_val_loaders(load_cached_data=False)

    print(f"    Train Loader Batches: {len(train_loader)}")
    print(f"    Val Loader Batches:   {len(val_loader)}")
    print(f"    File Size Stats (Mean, Std): {fs_stats}")

    # Verify batch structure
    try:
        images, file_sizes, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"    Sample Batch Shapes:")
    print(f"      Images:     {images.shape}")
    print(f"      File Sizes: {file_sizes.shape}")
    print(f"      Labels:     {labels.shape}")

    # Assertions to ensure data integrity
    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert file_sizes.shape == (Config.BATCH_SIZE,), "Incorrect file_size batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Models...")

    # Initialize all models to check they build correctly
    # Using smaller parameters for NeXt to initialize quickly
    models = {
        "RepVGG": CactusRepVGG(num_classes=1, deploy=False),
        "ResNet": CactusResNet(num_classes=1),
        "NeXt": CactusNeXt(num_classes=1, depths=[1, 1, 1, 1], dims=[16, 32, 64, 128]),
    }

    device = Config.DEVICE

    for name, model in models.items():
        print(f"    Testing {name} forward pass...")
        model.to(device)

        # Move batch to device
        img_batch = images.to(device)
        fs_batch = file_sizes.to(device)

        # Forward pass
        output = model(img_batch, fs_batch)

        # Verify output shape (Batch, Num_Classes)
        assert output.shape == (Config.BATCH_SIZE, 1), f"{name} output shape mismatch"
        print(f"      {name} Output: {output.shape} - OK")

        # Clean up to save memory
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (RepVGG)...")

    # Use RepVGG for the training demo
    model = models["RepVGG"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    # train_one_epoch iterates through the entire loader provided
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )

    print(f"    Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 5. Validation Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Validation...")

    val_loss, val_auc = validate(model, val_loader, criterion, device)

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation AUC:  {val_auc:.4f}")

    # Basic sanity checks
    assert val_loss >= 0, "Validation loss negative"
    assert 0.0 <= val_auc <= 1.0, "AUC out of range"

    # -------------------------------------------------------------------------
    # 6. SWA (Stochastic Weight Averaging) Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Testing SWA Handler...")

    # Initialize SWA to start immediately (swa_start_epoch=0)
    swa_handler = SWAHandler(
        model, optimizer, swa_start_epoch=0, swa_lr=1e-4, device=device
    )

    # Trigger SWA step (epoch 1 >= 0)
    step_result = swa_handler.step(epoch=1, model=model)
    print(f"    SWA Step Triggered: {step_result}")
    assert step_result is True, "SWA should have triggered"

    # Update BN statistics using the training loader
    print("    Updating SWA BN statistics...")
    swa_handler.update_bn(train_loader)

    swa_model = swa_handler.get_model()

    # Verify SWA model works
    with torch.no_grad():
        swa_out = swa_model(images.to(device), file_sizes.to(device))
    assert swa_out.shape == (Config.BATCH_SIZE, 1), "SWA model output mismatch"
    print("    SWA Model Forward Pass - OK")

    # -------------------------------------------------------------------------
    # 7. Inference with TTA
    # -------------------------------------------------------------------------
    print("\n[7] Testing Inference with TTA...")

    # Load test loader
    # Note: get_test_loader loads the full test set.
    test_loader, test_ids = get_test_loader(fs_stats, load_cached_data=False)

    # Perform inference (4-view TTA)
    # This might take a few seconds on CPU for ~3k images
    preds = inference_tta(model, test_loader, device)

    print(f"    Predictions Shape: {preds.shape}")
    print(f"    Sample Predictions: {preds[:5]}")

    assert len(preds) == len(test_ids), "Prediction count mismatch"
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range"

    # -------------------------------------------------------------------------
    # 8. RepVGG Deployment Transformation
    # -------------------------------------------------------------------------
    print("\n[8] Testing RepVGG Deployment Switch...")

    # Use a fresh model instance to clearly show the structural change
    deploy_demo_model = CactusRepVGG(num_classes=1, deploy=False)
    deploy_demo_model.eval()

    # Check for existence of training-time branches (e.g., rbr_dense) in the first block of stage 1
    # Accessing internal structure: stage1 is a Sequential, index 0 is RepVGGBlock
    block = deploy_demo_model.stage1[0]
    assert hasattr(
        block, "rbr_dense"
    ), "Model should have rbr_dense before deploy switch"

    print("    Switching to deploy mode...")
    deploy_demo_model.switch_to_deploy()

    # Check that training branches are removed and reparam branch exists
    assert not hasattr(
        block, "rbr_dense"
    ), "Model should NOT have rbr_dense after deploy switch"
    assert hasattr(
        block, "rbr_reparam"
    ), "Model should have rbr_reparam after deploy switch"

    # Verify forward pass still works
    with torch.no_grad():
        out_deploy = deploy_demo_model(images, file_sizes)
    assert out_deploy.shape == (Config.BATCH_SIZE, 1)
    print("    Deploy Model Forward Pass - OK")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
