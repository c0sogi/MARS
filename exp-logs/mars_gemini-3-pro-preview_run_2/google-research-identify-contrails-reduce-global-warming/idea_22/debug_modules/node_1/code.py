import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config, setup_system
from library.utils import set_seed, rle_encode, dice_score
from library.data import get_dataloader, load_data
from library.model import ExtendedConvNeXtUNet
from library.loss import FocalBatchDiceLoss
from library.engine import train_one_epoch, validate, inference


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print("Initializing system...")
    setup_system(seed=42)

    # Override Config for demonstration speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Define sample size for quick testing
    SAMPLE_SIZE = 8

    # Prepare a subset of test metadata for Inference consistency
    # (inference() reads the file directly, so we need a file with matching row count)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    original_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    subset_test_meta_path = os.path.join(Config.WORKING_DIR, "test_metadata_subset.csv")
    original_test_meta.head(SAMPLE_SIZE).to_csv(subset_test_meta_path, index=False)

    # Temporarily point Config to this subset
    ORIGINAL_TEST_PATH = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = subset_test_meta_path

    device = Config.DEVICE
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n[Demo] Data Loading...")
    # Load a small subset of training data
    # We use load_cached_data=False to ensure we test the processing logic
    train_loader = get_dataloader(
        split="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
        sample_size=SAMPLE_SIZE,
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    print(f"  Batch Image Shape: {images.shape}")  # Should be (B, 6, 256, 256)
    print(f"  Batch Mask Shape:  {masks.shape}")  # Should be (B, 1, 256, 256)

    # Validations
    if images.shape != (Config.BATCH_SIZE, 6, 256, 256):
        raise AssertionError(
            f"Expected image shape {(Config.BATCH_SIZE, 6, 256, 256)}, got {images.shape}"
        )
    if masks.shape != (Config.BATCH_SIZE, 1, 256, 256):
        raise AssertionError(
            f"Expected mask shape {(Config.BATCH_SIZE, 1, 256, 256)}, got {masks.shape}"
        )

    # ==========================================
    # 3. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[Demo] Model Instantiation...")
    model = ExtendedConvNeXtUNet(in_channels=Config.IN_CHANNELS, num_classes=1)
    model.to(device)

    print("  Running forward pass...")
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)

    print(f"  Output Logits Shape: {logits.shape}")

    if logits.shape != (Config.BATCH_SIZE, 1, 256, 256):
        raise AssertionError("Model output shape mismatch.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[Demo] Loss Calculation...")
    criterion = FocalBatchDiceLoss(gamma=2.0)
    masks = masks.to(device)

    loss = criterion(logits, masks)
    print(f"  Calculated Loss: {loss.item():.6f}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")

    # ==========================================
    # 5. Training Loop Component
    # ==========================================
    print("\n[Demo] Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Check weights before update
    head_weight_before = model.head.weight.data.clone()

    # Run one epoch on the subset loader
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"  Average Train Loss: {avg_loss:.6f}")

    # Check weights after update
    head_weight_after = model.head.weight.data
    if torch.equal(head_weight_before, head_weight_after):
        raise AssertionError("Model weights did not update after training step.")
    else:
        print("  Model weights updated successfully.")

    # ==========================================
    # 6. Validation Component
    # ==========================================
    print("\n[Demo] Validation Step...")
    # Create validation loader (subset)
    val_loader = get_dataloader(
        split="validation",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
        sample_size=SAMPLE_SIZE,
    )

    val_dice, val_loss = validate(model, val_loader, device)
    print(f"  Validation Dice: {val_dice:.6f}")
    print(f"  Validation Loss: {val_loss:.6f}")

    # ==========================================
    # 7. Utilities Verification
    # ==========================================
    print("\n[Demo] Utilities Verification...")

    # Test RLE Encoding
    # 3x3 mask, center pixel is 1. Flattened (col-major): 0,0,0, 0,1,0, 0,0,0 -> index 5 (1-based) is 1.
    dummy_mask = np.zeros((3, 3), dtype=np.uint8)
    dummy_mask[1, 1] = 1
    rle = rle_encode(dummy_mask)
    print(f"  RLE for 3x3 center pixel: '{rle}'")
    if rle != "5 1":
        raise AssertionError(f"RLE Encoding failed. Expected '5 1', got '{rle}'")

    # Test Dice Score
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 1]])  # 1 FP
    # Intersection = 2, Union = 3 + 2 = 5. Dice = 2*2 / 5 = 0.8
    dice = dice_score(y_pred, y_true)
    print(f"  Dice Score check: {dice:.4f}")
    if not (0.79 < dice < 0.81):
        raise AssertionError("Dice score calculation incorrect.")

    # ==========================================
    # 8. Inference Pipeline
    # ==========================================
    print("\n[Demo] Inference Pipeline...")

    # Create test loader
    # Note: We use the subset metadata file created earlier, so we don't pass sample_size
    # (load_data reads the full file, which is already a subset)
    test_loader = get_dataloader(
        split="test", batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Run inference
    inference(model, test_loader, device)

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission rows: {len(sub_df)}")
    print(f"  First few entries:\n{sub_df.head()}")

    if len(sub_df) != SAMPLE_SIZE:
        raise AssertionError(
            f"Expected {SAMPLE_SIZE} rows in submission, found {len(sub_df)}."
        )

    # Cleanup
    Config.TEST_METADATA_PATH = ORIGINAL_TEST_PATH
    if os.path.exists(subset_test_meta_path):
        os.remove(subset_test_meta_path)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
