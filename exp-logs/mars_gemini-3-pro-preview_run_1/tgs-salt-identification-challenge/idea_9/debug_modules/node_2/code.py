import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

# Import provided library components
from library.utils import seed_everything, rle_encode, rle_decode, calculate_iou_map
from library.dataset import SaltDataset
from library.model import DeepResUNet
from library.loss import CompoundLoss
from library.trainer import train_one_epoch, validate
from library.inference import generate_submission, predict_snapshot


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    print("=== Setting up environment ===")
    WORK_DIR = "./working/demo_execution"
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Verify Utils (RLE Encoding/Decoding)
    # -------------------------------------------------------------------------
    print("\n=== Verifying Utils ===")
    # Create a synthetic 101x101 mask with a known pattern (a 10x10 square)
    synthetic_mask = np.zeros((101, 101), dtype=np.uint8)
    synthetic_mask[10:20, 10:20] = 1

    # Encode
    rle_str = rle_encode(synthetic_mask)
    print(f"RLE String sample: {rle_str[:20]}...")

    # Decode
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    # Verify
    assert np.array_equal(synthetic_mask, decoded_mask), "RLE Decode mismatch!"
    print("RLE Encode/Decode verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n=== Verifying Dataset ===")
    # Initialize dataset
    full_train_dataset = SaltDataset(mode="train", work_dir=WORK_DIR)

    # Create a small subset for speed (16 samples)
    indices = list(range(16))
    train_subset = Subset(full_train_dataset, indices)

    # Create DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
    )

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Verify shapes
    # Images are padded to 128x128 in dataset.py
    print(
        f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    assert images.shape == (4, 1, 128, 128), "Incorrect image tensor shape"
    assert masks.shape == (4, 1, 128, 128), "Incorrect mask tensor shape"
    assert depths.shape == (4, 1), "Incorrect depth tensor shape"

    # Verify value ranges
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Image values out of range [0, 1]"
    assert set(np.unique(masks.numpy()).tolist()).issubset(
        {0.0, 1.0}
    ), "Mask values must be binary (0 or 1)"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model and Loss
    # -------------------------------------------------------------------------
    print("\n=== Verifying Model and Loss ===")
    model = DeepResUNet(in_channels=1, out_channels=1).to(device)
    criterion = CompoundLoss().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward pass (Training mode)
    model.train()
    outputs = model(images, depths)

    # Model returns (final, aux1, aux2) in training mode
    assert (
        len(outputs) == 3
    ), "Model should return 3 outputs in training mode (Deep Supervision)"
    final_out, aux1, aux2 = outputs

    assert final_out.shape == (4, 1, 128, 128), "Output shape mismatch"

    # Calculate Loss
    loss = criterion(final_out, masks)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop (One Epoch)
    # -------------------------------------------------------------------------
    print("\n=== Verifying Training Loop ===")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Check weights before update
    param_before = next(model.parameters()).clone()

    # Run one epoch using the library function
    # Note: train_one_epoch iterates over the loader provided
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch Average Loss: {avg_loss:.4f}")

    # Check weights after update
    param_after = next(model.parameters())
    assert not torch.equal(param_before, param_after), "Model weights did not update!"

    # Verify Validation
    # We use the same subset for validation just to check the function works
    val_score = validate(model, train_loader, device)
    print(f"Validation mAP: {val_score:.4f}")
    assert 0.0 <= val_score <= 1.0, "mAP score out of range"

    print("Training loop verification passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Inference and Submission
    # -------------------------------------------------------------------------
    print("\n=== Verifying Inference ===")

    # Save the current model as a checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Generate submission
    # This function loads the test set (from metadata/test.csv), runs inference, and saves CSV
    # We use the checkpoint we just saved.
    sub_path = os.path.join(WORK_DIR, "predictions", "submission.csv")

    # Note: generate_submission uses SaltDataset(mode='test'), which reads metadata/test.csv.
    # It processes all 1000 test images. This is reasonably fast.
    generate_submission(
        snapshot_paths=[ckpt_path],
        work_dir=WORK_DIR,
        output_path=sub_path,
        batch_size=16,  # Smaller batch size for safety
        device_name="cuda" if torch.cuda.is_available() else "cpu",
        load_cached_data=False,  # Force reload to ensure clean state
    )

    # Verify submission file
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    # Ensure rle_mask is strictly string, handling Pandas NaN for empty fields (Cite debug_lesson_8)
    df_sub["rle_mask"] = df_sub["rle_mask"].fillna("")
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission missing required columns"

    # Check row count (Test set has 1000 images)
    assert len(df_sub) == 1000, f"Expected 1000 rows in submission, got {len(df_sub)}"

    # Check content format (RLE should be string or NaN if empty, but code produces empty string usually)
    # The rle_encode function returns a string.
    sample_rle = df_sub.iloc[0]["rle_mask"]
    assert isinstance(sample_rle, str), "RLE mask should be a string"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
