import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import provided library components
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    calculate_iou_batch,
    calculate_map,
)
from library.dataset import get_dataloaders
from library.model import ResUNetASPP
from library.losses import Phase1Loss, Phase2Loss
from library.engine import train_one_epoch, evaluate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Salt Segmentation Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast execution...")

    # Override Config for a quick demo run
    Config.DEBUG = True
    Config.DEBUG_SIZE = 32  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.PHASE_1_EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTIONS_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"   Device: {device}")
    print(f"   Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE & Metrics)
    # -------------------------------------------------------------------------
    print("\n2. Verifying Utility Functions...")

    # Test RLE Encode/Decode
    # Create a 101x101 mask with a 10x10 square of 1s at (10,10)
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE decoded mask does not match original"
    print("   [PASS] RLE Encode/Decode consistency check.")

    # Test IoU Calculation
    # Case 1: Perfect match
    iou_perfect = calculate_iou_batch(dummy_mask[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        iou_perfect, 1.0
    ), f"Perfect match IoU should be 1.0, got {iou_perfect}"

    # Case 2: No overlap
    dummy_mask_2 = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask_2[50:60, 50:60] = 1
    iou_zero = calculate_iou_batch(dummy_mask[None, ...], dummy_mask_2[None, ...])
    assert np.isclose(iou_zero, 0.0), f"No overlap IoU should be 0.0, got {iou_zero}"

    # Test mAP Calculation
    # Perfect match should give mAP 1.0 across all thresholds
    map_score = calculate_map(dummy_mask[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        map_score, 1.0
    ), f"Perfect match mAP should be 1.0, got {map_score}"
    print("   [PASS] IoU and mAP metric calculation check.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n3. Initializing Data Pipeline...")

    # Load dataloaders (this will trigger caching logic in library.dataset)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached=False
    )

    print(f"   Train Batches: {len(train_loader)}")
    print(f"   Val Batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, masks, ids = next(iter(train_loader))

    # Expected shapes:
    # Images: (B, 2, 128, 128) -> 2 channels (Gray + Depth), Padded to 128
    # Masks: (B, 1, 128, 128) -> Padded to 128
    assert images.shape == (
        Config.BATCH_SIZE,
        2,
        128,
        128,
    ), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {masks.shape}"

    # Verify Depth Fusion
    # The second channel (index 1) should be the depth channel.
    # It should be constant for a single image (spatial dimensions).
    depth_channel = images[0, 1, :, :].numpy()
    assert np.all(
        depth_channel == depth_channel[0, 0]
    ), "Depth channel is not constant spatially"
    print("   [PASS] DataLoader shapes and depth fusion verification.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n4. Instantiating Model and Verifying Forward Pass...")

    model = ResUNetASPP(
        in_channels=Config.IN_CHANNELS, out_channels=Config.OUT_CHANNELS
    )
    model = model.to(device)

    images = images.to(device)

    # Test Training Mode (Deep Supervision)
    model.train()
    outputs = model(images)
    assert len(outputs) == 3, "Model in train mode should return (logits, aux2, aux3)"
    logits, aux2, aux3 = outputs
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Logits shape mismatch: {logits.shape}"
    assert aux2.shape == (Config.BATCH_SIZE, 1, 128, 128), "Aux2 shape mismatch"
    print("   [PASS] Model training forward pass (Deep Supervision).")

    # Test Eval Mode
    model.eval()
    with torch.no_grad():
        logits_eval = model(images)
    assert isinstance(
        logits_eval, torch.Tensor
    ), "Model in eval mode should return single tensor"
    assert logits_eval.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Eval logits shape mismatch"
    print("   [PASS] Model evaluation forward pass.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n5. Verifying Loss Functions...")

    masks = masks.to(device)

    # Phase 1 Loss (BCE + Dice)
    criterion_p1 = Phase1Loss()
    loss_p1 = criterion_p1(logits, masks)
    assert not torch.isnan(loss_p1), "Phase 1 loss returned NaN"
    assert loss_p1.item() > 0, "Phase 1 loss should be positive"

    # Phase 2 Loss (BCE + Lovasz)
    criterion_p2 = Phase2Loss()
    loss_p2 = criterion_p2(logits, masks)
    assert not torch.isnan(loss_p2), "Phase 2 loss returned NaN"

    print(f"   Phase 1 Loss (Sample): {loss_p1.item():.4f}")
    print(f"   Phase 2 Loss (Sample): {loss_p2.item():.4f}")
    print("   [PASS] Loss function execution.")

    # -------------------------------------------------------------------------
    # 6. Training & Evaluation Loop Demo
    # -------------------------------------------------------------------------
    print("\n6. Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch using the engine
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"   Epoch 0 Training Loss: {train_loss:.4f}")

    # Run evaluation
    print("   Running Evaluation...")
    val_loss, val_map = evaluate(model, val_loader, device, epoch=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation mAP: {val_map:.4f}")

    assert val_map >= 0.0 and val_map <= 1.0, "mAP score out of range"
    print("   [PASS] Training and Evaluation loop completed.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n7. Generating Submission...")

    output_csv = Config.SUBMISSION_PATH
    generate_submission(model, test_loader, device, output_csv)

    assert os.path.exists(output_csv), "Submission file was not created"

    df_sub = pd.read_csv(output_csv)
    print(f"   Submission file created at: {output_csv}")
    print(f"   Rows: {len(df_sub)}")
    print(f"   Columns: {list(df_sub.columns)}")

    # Check format of first mask
    first_rle = df_sub.iloc[0]["rle_mask"]
    if pd.notna(first_rle) and first_rle != "":
        # Try decoding to ensure validity
        try:
            rle_decode(first_rle)
            print("   [PASS] Submission RLE format verified.")
        except Exception as e:
            print(f"   [FAIL] Invalid RLE in submission: {e}")
            raise
    else:
        print("   [INFO] First prediction was empty (valid for empty masks).")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
