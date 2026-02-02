import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rle_encode, dice_score
from library.dataset import ContrailDataset, get_dataloader
from library.model import ContrailUNet
from library.loss import DiceBCELoss
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import make_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Contrail Identification Library Demo ===\n")

    # 1. Setup & Configuration
    # ------------------------
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Override Config for speed in this demo
    Config.DEBUG = True
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small demo

    # Define working paths
    demo_dir = os.path.join(Config.OUTPUT_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)
    checkpoint_path = os.path.join(demo_dir, "demo_model.pth")
    submission_path = os.path.join(demo_dir, "submission.csv")

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Verify Utilities
    # -------------------
    print("\n[1/5] Verifying Utilities...")

    # Test RLE Encode
    # Create a 2x2 mask: [[0, 1], [0, 0]]
    # Flattened (Column-major 'F'): [0, 0, 1, 0] -> Index 3 (1-based) is 1.
    dummy_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    rle = rle_encode(dummy_mask)
    print(f"  RLE Input (2x2): [[0, 1], [0, 0]] -> RLE Output: '{rle}'")
    assert rle == "3 1", f"RLE failed. Expected '3 1', got '{rle}'"

    # Test Dice Score
    pred_t = torch.tensor([1.0, 1.0, 0.0])
    target_t = torch.tensor([1.0, 0.0, 1.0])
    # Intersection: 1 (first element)
    # Union: 2 + 2 = 4
    # Dice: (2*1 + smooth) / (4 + smooth) approx 0.5
    score = dice_score(pred_t, target_t, threshold=0.5, smooth=0.0)
    print(
        f"  Dice Score Check: Pred={pred_t.numpy()}, Target={target_t.numpy()} -> Dice={score:.4f}"
    )
    assert abs(score - 0.5) < 1e-5, "Dice score calculation incorrect"
    print("  Utilities verified.")

    # 3. Verify Dataset & DataLoader
    # ------------------------------
    print("\n[2/5] Verifying Dataset & DataLoader...")

    # Initialize dataset with max_samples limit for speed
    train_ds = ContrailDataset(split="train", debug=True, max_samples=10)
    print(f"  Train Dataset Size (Debug): {len(train_ds)}")

    if len(train_ds) > 0:
        sample = train_ds[0]
        image = sample["image"]
        mask = sample["mask"]
        record_id = sample["record_id"]

        print(f"  Sample Record ID: {record_id}")
        print(f"  Image Shape: {image.shape} (Expected: (9, 256, 256))")
        print(f"  Mask Shape: {mask.shape} (Expected: (1, 256, 256))")

        # Assertions
        assert image.shape == (9, 256, 256), "Incorrect image dimensions"
        assert mask.shape == (1, 256, 256), "Incorrect mask dimensions"
        assert isinstance(image, torch.Tensor), "Image is not a Tensor"
    else:
        print("  Warning: Dataset is empty. Skipping shape checks.")

    # Create DataLoader
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, debug=True
    )
    # We iterate once to ensure it works
    batch = next(iter(train_loader))
    print(f"  Batch Image Shape: {batch['image'].shape}")
    print("  Dataset & DataLoader verified.")

    # 4. Verify Model
    # ---------------
    print("\n[3/5] Verifying Model Architecture...")
    model = ContrailUNet()
    model.to(device)

    # Create dummy input: Batch=2, Channels=9, H=256, W=256
    dummy_input = torch.randn(2, 9, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    assert output.shape == (2, 1, 256, 256), "Model output shape mismatch"
    print("  Model verified.")

    # 5. Training & Validation Loop Demo
    # ----------------------------------
    print("\n[4/5] Running Training & Validation Loop (1 Epoch)...")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train
    print("  Training...")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=1)
    print(f"  Train Loss: {train_loss:.6f}")

    # Validate
    # Use a small validation set
    val_loader = get_dataloader(
        split="validation", batch_size=Config.BATCH_SIZE, debug=True
    )
    print("  Validating...")
    val_loss, val_dice = valid_one_epoch(model, val_loader, device)
    print(f"  Val Loss: {val_loss:.6f} | Global Dice: {val_dice:.6f}")

    # Save Model for Inference Step
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Model saved to {checkpoint_path}")

    # 6. Inference & Submission
    # -------------------------
    print("\n[5/5] Running Inference & Submission...")

    # We use the checkpoint we just saved
    # Note: make_submission internally re-loads the model and creates a test dataloader
    # We enable debug=True to run on a subset of the test set

    try:
        make_submission(
            checkpoint_path=checkpoint_path,
            output_csv=submission_path,
            batch_size=Config.BATCH_SIZE,
            device=device,
            debug=True,
        )

        # Verify output
        if os.path.exists(submission_path):
            df_sub = pd.read_csv(submission_path)
            print(f"  Submission file created at {submission_path}")
            print(f"  Number of predictions: {len(df_sub)}")
            print(f"  Columns: {list(df_sub.columns)}")

            # Check format
            assert "record_id" in df_sub.columns and "encoded_pixels" in df_sub.columns
            print("  Submission format verified.")
        else:
            raise FileNotFoundError("Submission file was not created.")

    except Exception as e:
        print(f"  Inference failed: {e}")
        raise e

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
