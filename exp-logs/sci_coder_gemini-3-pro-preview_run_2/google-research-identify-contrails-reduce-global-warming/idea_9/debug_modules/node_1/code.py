import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coef_metric
from library.dataset import get_dataloader, ContrailDataset
from library.model import HybridResNetTransformerUNet
from library.loss import HybridLoss
from library.train import train_model
from library.predict import predict_and_submit


def run_demo():
    print("Starting Contrail Detection Demo...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    # Override Config defaults to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set hyperparameters for quick execution
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.TTA_ENABLED = False  # Disable TTA for speed

    # Define sample limits
    DEMO_TRAIN_SAMPLES = 4
    DEMO_VAL_SAMPLES = 4
    DEMO_TEST_SAMPLES = 4

    print(f"Configuration updated.")
    print(f"  Working Dir: {Config.WORKING_DIR}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Utility Verification
    # ==========================================
    print("\nVerifying Utilities...")

    # Test RLE Encoding
    # Create a 4x4 mask
    # 0 1 0 0
    # 0 1 0 0
    # 0 0 0 0
    # 0 0 0 0
    # Flattened (column-major): 0,0,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0
    # Indices (1-based): 5, 6
    # Run: start 5, length 2
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0:2, 1] = 1
    encoded = rle_encode(dummy_mask)
    expected_rle = "5 2"
    assert (
        encoded == expected_rle
    ), f"RLE Encoding failed. Expected '{expected_rle}', got '{encoded}'"
    print("  RLE Encoding verified.")

    # Test Dice Metric
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_pred_perfect = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_pred_wrong = torch.tensor([0.0, 0.0, 1.0, 1.0])

    dice_perfect = dice_coef_metric(y_pred_perfect, y_true, threshold=0.5)
    dice_wrong = dice_coef_metric(y_pred_wrong, y_true, threshold=0.5)

    assert np.isclose(
        dice_perfect, 1.0
    ), f"Dice metric failed for perfect match. Got {dice_perfect}"
    assert np.isclose(
        dice_wrong, 0.0
    ), f"Dice metric failed for mismatch. Got {dice_wrong}"
    print("  Dice Metric verified.")

    # ==========================================
    # 3. Dataset & DataLoader Verification
    # ==========================================
    print("\nVerifying Dataset and DataLoader...")

    # Instantiate Dataset
    train_ds = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=None,  # No transform for shape check
        max_samples=DEMO_TRAIN_SAMPLES,
    )

    assert (
        len(train_ds) == DEMO_TRAIN_SAMPLES
    ), f"Dataset length mismatch. Expected {DEMO_TRAIN_SAMPLES}, got {len(train_ds)}"

    # Check item shape
    img, mask = train_ds[0]
    # Image: (C, H, W) -> (6, 256, 256)
    # Mask: (C, H, W) -> (1, 256, 256)
    assert img.shape == (6, 256, 256), f"Image shape incorrect. Got {img.shape}"
    assert mask.shape == (1, 256, 256), f"Mask shape incorrect. Got {mask.shape}"
    assert img.dtype == torch.float32, "Image dtype should be float32"
    assert mask.dtype == torch.float32, "Mask dtype should be float32"

    # Check DataLoader
    train_loader = get_dataloader(
        split="train",
        batch_size=Config.BATCH_SIZE,
        max_samples=DEMO_TRAIN_SAMPLES,
        num_workers=0,
    )
    batch_imgs, batch_masks = next(iter(train_loader))

    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        6,
        256,
        256,
    ), f"Batch image shape incorrect: {batch_imgs.shape}"
    assert batch_masks.shape == (
        Config.BATCH_SIZE,
        1,
        256,
        256,
    ), f"Batch mask shape incorrect: {batch_masks.shape}"
    print("  Dataset and DataLoader verified.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\nVerifying Model Architecture...")

    model = HybridResNetTransformerUNet()
    model.to(Config.DEVICE)
    model.eval()

    # Forward pass with dummy batch
    with torch.no_grad():
        dummy_input = torch.randn(2, 6, 256, 256).to(Config.DEVICE)
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape incorrect. Expected (2, 1, 256, 256), got {output.shape}"
    print("  Model forward pass verified.")

    # ==========================================
    # 5. Loss Function Verification
    # ==========================================
    print("\nVerifying Loss Function...")

    criterion = HybridLoss()
    dummy_logits = torch.randn(2, 1, 256, 256)  # Random logits
    dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Binary targets

    loss = criterion(dummy_logits, dummy_targets)

    assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"  HybridLoss calculation verified. Value: {loss.item():.4f}")

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\nRunning Training Loop (1 Epoch, Tiny Subset)...")

    # We use the train_model function from library.train
    # It handles loader creation, training, validation, and saving
    try:
        train_model(
            max_train_samples=DEMO_TRAIN_SAMPLES,
            max_val_samples=DEMO_VAL_SAMPLES,
            epochs=Config.EPOCHS,
        )
    except Exception as e:
        raise AssertionError(f"Training loop failed: {e}")

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise AssertionError(f"Model file was not saved to {Config.MODEL_PATH}")

    print(f"  Training verified. Model saved to {Config.MODEL_PATH}")

    # ==========================================
    # 7. Inference Demonstration
    # ==========================================
    print("\nRunning Inference and Submission Generation...")

    try:
        predict_and_submit(max_samples=DEMO_TEST_SAMPLES)
    except Exception as e:
        raise AssertionError(f"Inference failed: {e}")

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_FILE}")

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    expected_cols = ["record_id", "encoded_pixels"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check length
    assert (
        len(df_sub) == DEMO_TEST_SAMPLES
    ), f"Submission length mismatch. Expected {DEMO_TEST_SAMPLES}, got {len(df_sub)}"

    # Check content format (record_id should be int/str, encoded_pixels str)
    assert not df_sub["record_id"].isnull().any(), "Submission contains null record_ids"
    assert (
        not df_sub["encoded_pixels"].isnull().any()
    ), "Submission contains null encoded_pixels"

    print(f"  Inference verified. Submission saved to {Config.SUBMISSION_FILE}")
    print("  Sample Submission Rows:")
    print(df_sub.head())

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
