import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_srm_kernels, weighted_auc, read_image
from library.dataset import (
    SteganalysisDataset,
    get_dataloaders,
    get_test_dataloader,
    get_transforms,
)
from library.model import SRMEfficientNet
from library.engine import train_one_epoch, validate, predict_tta


def run_demo():
    print("=== Starting ALASKA2 Steganalysis Library Demo ===\n")

    # --- 1. Setup and Configuration ---
    print("[1] Setting up Configuration and Seeding...")
    seed_everything(42)

    # Create a temporary working directory for this demo
    demo_working_dir = "./working/demo_run"
    os.makedirs(demo_working_dir, exist_ok=True)

    # Update Config for speed:
    # - Enable DEBUG mode to sample a tiny subset of data
    # - Reduce Epochs and Batch Size
    # - Disable pretrained weights to avoid download overhead during demo
    # - Point to a custom test metadata file (created later)
    Config.update(
        DEBUG=True,
        DEBUG_SAMPLE_SIZE=20,  # Only use 20 images for train/val
        BATCH_SIZE=4,
        EPOCHS=1,
        WORKING_DIR=demo_working_dir,
        CHECKPOINT_PATH=os.path.join(demo_working_dir, "demo_model.pth"),
        SUBMISSION_PATH=os.path.join(demo_working_dir, "demo_submission.csv"),
    )

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print("    Configuration updated for fast demonstration.\n")

    # --- 2. Verify Utilities ---
    print("[2] Verifying Utilities...")

    # Test SRM Kernels
    kernels = get_srm_kernels()
    print(f"    SRM Kernels shape: {kernels.shape}")
    assert kernels.shape == (30, 1, 5, 5), "SRM Kernels should be (30, 1, 5, 5)"
    assert kernels.dtype == torch.float32, "SRM Kernels should be float32"

    # Test Weighted AUC
    # Case 1: Perfect prediction
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    score_perfect = weighted_auc(y_true, y_pred_perfect)
    print(f"    Weighted AUC (Perfect): {score_perfect:.4f}")
    assert score_perfect == 1.0, "Perfect prediction should yield AUC 1.0"

    # Case 2: Random/Bad prediction
    y_pred_bad = np.array([0.9, 0.8, 0.2, 0.1])
    score_bad = weighted_auc(y_true, y_pred_bad)
    print(f"    Weighted AUC (Inverse): {score_bad:.4f}")
    assert score_bad == 0.0, "Inverse prediction should yield AUC 0.0"
    print("    Utilities verification passed.\n")

    # --- 3. Verify Dataset and DataLoaders ---
    print("[3] Verifying Dataset and DataLoaders...")

    # Get dataloaders in debug mode
    train_loader, val_loader = get_dataloaders(debug=True)

    print(f"    Train Loader length: {len(train_loader)} batches")
    print(f"    Val Loader length: {len(val_loader)} batches")

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    # Expected shape: (Batch_Size, 1, 512, 512) because we extract Y channel and add dim
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), f"Expected image shape {(Config.BATCH_SIZE, 1, 512, 512)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # Check value range (should be normalized [0, 1])
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images should be normalized to [0, 1]"
    print("    Dataset verification passed.\n")

    # --- 4. Verify Model Architecture ---
    print("[4] Verifying Model Architecture...")

    # Instantiate model (pretrained=False for speed)
    model = SRMEfficientNet(backbone_name="efficientnet_b0", pretrained=False)
    model = model.to(device)

    # Forward pass with dummy data
    dummy_input = torch.randn(2, 1, 512, 512).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("    Model verification passed.\n")

    # --- 5. Verify Training and Validation Loop ---
    print("[5] Verifying Training and Validation Engine...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch
    print("    Running training step...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    print("    Running validation step...")
    val_loss, val_score = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f} | Val Weighted AUC: {val_score:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_score <= 1.0, "Validation score out of range [0, 1]"
    print("    Engine verification passed.\n")

    # --- 6. Verify Inference (TTA) ---
    print("[6] Verifying Inference with TTA...")

    # Create a mini test set metadata file to avoid processing 5000 images
    test_full_df = pd.read_csv(Config.TEST_METADATA)
    test_mini_df = test_full_df.head(5).copy()
    mini_test_path = os.path.join(demo_working_dir, "test_mini.csv")
    test_mini_df.to_csv(mini_test_path, index=False)

    # Update config to use this mini test set
    Config.update(TEST_METADATA=mini_test_path)

    # Get test loader
    test_loader = get_test_dataloader()
    assert test_loader is not None, "Test loader should not be None"

    # Run prediction
    print("    Running predict_tta on 5 test images...")
    ids, preds = predict_tta(model, test_loader, device)

    print(f"    Predictions count: {len(preds)}")
    print(f"    Sample IDs: {ids[:2]}")
    print(f"    Sample Preds: {preds[:2]}")

    assert len(ids) == 5, "Should have 5 IDs"
    assert len(preds) == 5, "Should have 5 predictions"
    assert np.all(
        (preds >= 0.0) & (preds <= 1.0)
    ), "Predictions should be probabilities [0, 1]"

    # Save submission (as per engine logic)
    sub_df = pd.DataFrame({"Id": ids, "Label": preds})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print("    Inference verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
