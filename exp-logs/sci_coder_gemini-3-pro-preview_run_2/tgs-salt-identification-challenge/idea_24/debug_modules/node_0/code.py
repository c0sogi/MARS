import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode, pad_image, unpad_image
from library.dataset import get_dataloaders
from library.model import ResNet34WideLinkNet
from library.losses import MultiTaskLoss
from library.engine import set_seed, train_one_epoch, evaluate, generate_submission


def run_demo():
    print("--- Starting Library Verification & Demo ---")

    # 1. Setup & Configuration Overrides for Demo
    # We modify Config directly to affect global behavior for this run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample for speed
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.EPOCHS = 1
    Config.IDEA_NAME = "demo_run"
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_NAME)
    Config.CACHE_DIR = os.path.join(Config.IDEA_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.IDEA_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")

    # Create directories
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Verify Utility Functions
    print("\n[1/5] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[1:4, 1:4] = 1  # A 3x3 square
    rle = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle, shape=(10, 10))

    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError(
            "RLE Encode -> Decode failed to reconstruct original mask."
        )
    print("  - RLE Encode/Decode: OK")

    # Test Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded_img = pad_image(dummy_img, target_h=128, target_w=128)

    if padded_img.shape != (128, 128):
        raise AssertionError(
            f"Padding failed. Expected (128, 128), got {padded_img.shape}"
        )

    unpadded_img = unpad_image(padded_img, original_h=101, original_w=101)

    if not np.array_equal(dummy_img, unpadded_img):
        raise AssertionError("Pad -> Unpad failed to reconstruct original image.")
    print("  - Pad/Unpad: OK")

    # 3. Verify Data Loading
    print("\n[2/5] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        load_cached_data=False,  # Force re-creation for demo
    )

    # Check Train Batch
    images, masks, depths = next(iter(train_loader))

    # Expected shapes:
    # Images: (B, 1, 128, 128) -> 1 channel because dataset converts grayscale
    # Masks: (B, 1, 128, 128)
    # Depths: (B, 1)

    if images.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Train Image batch shape mismatch: {images.shape}")
    if masks.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Train Mask batch shape mismatch: {masks.shape}")
    if depths.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(f"Train Depth batch shape mismatch: {depths.shape}")

    print(
        f"  - Train Loader Batch Shapes: Images={images.shape}, Masks={masks.shape}, Depths={depths.shape}"
    )
    print("  - Data Loading: OK")

    # 4. Verify Model & Loss
    print("\n[3/5] Verifying Model & Loss...")
    model = ResNet34WideLinkNet(pretrained=False).to(
        device
    )  # No pretrained weights download for speed/offline
    criterion = MultiTaskLoss()

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward Pass
    logits, pred_depths = model(images)

    if logits.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Model Output Logits shape mismatch: {logits.shape}")
    if pred_depths.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(f"Model Output Depth shape mismatch: {pred_depths.shape}")

    # Loss Calculation
    loss, metrics = criterion(logits, pred_depths, masks, depths)

    if not torch.isfinite(loss):
        raise AssertionError("Loss is not finite (NaN or Inf).")

    print(f"  - Forward Pass Loss: {loss.item():.4f}")
    print(f"  - Loss Components: {metrics}")
    print("  - Model & Loss: OK")

    # 5. Verify Training & Evaluation Loop
    print("\n[4/5] Running Training & Evaluation Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train 1 Epoch
    train_loss, train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )

    # Evaluate
    val_loss, val_map = evaluate(model, val_loader, criterion, device)

    print(f"  - Train Loss: {train_loss:.4f}")
    print(f"  - Val Loss: {val_loss:.4f}")
    print(f"  - Val mAP: {val_map:.4f}")

    if not (0 <= val_map <= 1):
        raise AssertionError(f"mAP score {val_map} is out of range [0, 1]")
    print("  - Training & Evaluation: OK")

    # 6. Verify Submission Generation
    print("\n[5/5] Generating Submission...")

    # Save best model (simulated)
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)

    # Run inference
    generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    # Check file content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    if list(df_sub.columns) != ["id", "rle_mask"]:
        raise AssertionError(
            f"Submission columns mismatch. Got: {list(df_sub.columns)}"
        )

    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"  - Submission saved to {Config.SUBMISSION_PATH}")
    print(f"  - Rows generated: {len(df_sub)}")
    print("  - Submission Generation: OK")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
