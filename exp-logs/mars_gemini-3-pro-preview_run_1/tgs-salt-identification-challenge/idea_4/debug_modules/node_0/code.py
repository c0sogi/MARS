import os
import torch
import numpy as np
import pandas as pd
import cv2

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import rle_encode, rle_decode, pad_image, crop_image
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.losses import DeepSupervisionLoss, BCEDiceLoss
from library.metrics import calculate_iou_map
from library.predict import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 32  # Small subset for quick execution
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.DEEP_SUPERVISION = True

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, BATCH_SIZE=8")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions (RLE, Padding, Cropping)...")

    # Test RLE Encoding/Decoding
    # Create a simple 101x101 mask with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    if not np.array_equal(dummy_mask, decoded):
        raise AssertionError("RLE Encode -> Decode cycle failed: Masks do not match.")
    print("RLE Encode/Decode verification passed.")

    # Test Padding and Cropping
    # Pad 101x101 -> 128x128
    padded = pad_image(dummy_mask, target_size=128)
    if padded.shape != (128, 128):
        raise AssertionError(f"Padding failed. Expected (128, 128), got {padded.shape}")

    # Crop 128x128 -> 101x101
    cropped = crop_image(padded, original_size=101)

    # Note: pad_image uses reflection padding.
    # Since our dummy mask has 0s at borders, reflection should preserve the 0s.
    # The center region (where we put 1s) should be untouched.
    if not np.array_equal(dummy_mask, cropped):
        raise AssertionError("Pad -> Crop cycle failed: Masks do not match.")
    print("Image Padding/Cropping verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading Data via get_dataloaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        load_cached_data=False,  # Force reload to demonstrate processing
    )

    # Fetch one batch from training loader
    images, masks = next(iter(train_loader))

    # Verify shapes
    # Images: (B, 2, 128, 128) -> 2 channels: Seismic + Depth
    # Masks: (B, 1, 128, 128)
    print(f"Loaded batch shapes - Images: {images.shape}, Masks: {masks.shape}")

    if images.shape != (Config.BATCH_SIZE, 2, 128, 128):
        raise AssertionError(f"Incorrect image batch shape: {images.shape}")
    if masks.shape != (Config.BATCH_SIZE, 1, 128, 128):
        raise AssertionError(f"Incorrect mask batch shape: {masks.shape}")

    # Verify Depth Channel
    # The second channel should be constant per image (depth value)
    depth_channel = images[0, 1, :, :].numpy()
    if not np.allclose(depth_channel, depth_channel[0, 0]):
        raise AssertionError("Depth channel is not constant across spatial dimensions.")
    print("Data Loading and Shape verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 4] Instantiating DeepResUNet and running Forward Pass...")

    device = Config.DEVICE
    model = DeepResUNet(in_channels=2, out_channels=1, deep_supervision=True)
    model = model.to(device)
    model.train()  # Set to train mode to enable Deep Supervision outputs

    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    outputs = model(images)

    # With Deep Supervision, output should be a list of tensors
    if not isinstance(outputs, list):
        raise AssertionError(
            "Model did not return a list despite deep_supervision=True"
        )

    print(f"Model returned {len(outputs)} outputs (Deep Supervision scales).")

    # Verify output shapes (High Res, Med Res, Low Res)
    # 128x128, 64x64, 32x32
    expected_shapes = [128, 64, 32]
    for i, out in enumerate(outputs):
        h, w = out.shape[2], out.shape[3]
        if h != expected_shapes[i] or w != expected_shapes[i]:
            raise AssertionError(
                f"Output {i} shape mismatch. Expected {expected_shapes[i]}, got {h}"
            )

    print("Model Forward Pass verification passed.")

    # -------------------------------------------------------------------------
    # 5. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Calculating DeepSupervisionLoss...")

    criterion = DeepSupervisionLoss()
    loss = criterion(outputs, masks)

    print(f"Calculated Loss: {loss.item():.4f}")

    # Verify backward pass capability
    loss.backward()
    print("Backward pass executed successfully.")

    # -------------------------------------------------------------------------
    # 6. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Metric (IoU mAP)...")

    # Use the highest resolution output for metric
    y_pred = torch.sigmoid(outputs[0])

    # Calculate metric on this batch
    # Note: calculate_iou_map expects probabilities or logits?
    # The docstring says "Predicted probabilities or binary masks".
    # We passed sigmoid probabilities.
    score = calculate_iou_map(y_pred, masks)

    print(f"Batch mAP Score: {score:.4f}")
    if not (0.0 <= score <= 1.0):
        raise AssertionError("mAP score out of range [0, 1]")

    # -------------------------------------------------------------------------
    # 7. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Inference Pipeline (generate_submission)...")

    # We save a dummy checkpoint because generate_submission loads it
    checkpoint_path = Config.CHECKPOINT_PATH
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved temporary checkpoint to {checkpoint_path}")

    # Run generation in debug mode
    generate_submission(debug=True)

    # Verify submission file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty.")

    print("Inference Pipeline verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
