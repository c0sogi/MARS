import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, do_kaggle_metric
from library.dataset import get_dataloaders, get_test_loader
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.engine import SaltEngine


def main():
    # =========================================================================
    # 1. Setup and Configuration Override
    # =========================================================================
    print(">>> Step 1: Setup and Configuration Override")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for a fast demonstration run
    print("Overriding Config parameters for demo speed...")
    Config.DEBUG_SAMPLES = 50  # Only use 50 samples for train/val
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data
    Config.TOTAL_EPOCHS = 2  # Run only 2 epochs total
    Config.PHASE1_EPOCHS = 1  # Switch to Phase 2 after epoch 0
    Config.WORK_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")

    # Create directories
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n>>> Step 2: Verifying Utility Functions (RLE & Metrics)")

    # Test RLE Encoding/Decoding
    # Create a simple 3x3 mask:
    # [[0, 1, 0],
    #  [0, 1, 0],
    #  [0, 1, 0]]
    # Flattened (column-major/Fortran): 0,0,0, 1,1,1, 0,0,0 -> indices 4,5,6 (1-based)
    dummy_mask = np.zeros((3, 3), dtype=np.uint8)
    dummy_mask[:, 1] = 1

    encoded = rle_encode(dummy_mask)
    print(f"Encoded RLE: '{encoded}'")

    # Expected: start 4, length 3
    assert encoded == "4 3", f"RLE Encoding failed. Expected '4 3', got '{encoded}'"

    decoded = rle_decode(encoded, shape=(3, 3))
    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE Decoding failed to recover original mask."
    print("RLE Encode/Decode logic verified.")

    # Test Kaggle Metric (IoU Precision)
    # Case 1: Perfect Match
    score_perfect = do_kaggle_metric(dummy_mask, dummy_mask, threshold=0.5)
    assert score_perfect == 1.0, f"Metric failed for perfect match. Got {score_perfect}"

    # Case 2: No Overlap
    dummy_pred_fail = np.zeros((3, 3), dtype=np.uint8)
    dummy_pred_fail[:, 0] = 1  # Different column
    score_fail = do_kaggle_metric(dummy_pred_fail, dummy_mask, threshold=0.5)
    assert score_fail == 0.0, f"Metric failed for no overlap. Got {score_fail}"
    print("Metric logic verified.")

    # =========================================================================
    # 3. Data Loading Demonstration
    # =========================================================================
    print("\n>>> Step 3: Initializing Data Loaders")

    # Get Train and Val loaders (Fold 0)
    # This will trigger caching in ./working/demo_execution/cache
    train_loader, val_loader = get_dataloaders(
        fold=0, load_cached_data=False, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, masks, ids = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 3, 128, 128)
    print(f"Batch Mask Shape: {masks.shape}")  # Should be (B, 128, 128)

    # Verify Input Channels (Seismic, Seismic, Depth)
    assert images.shape[1] == 3, "Input should have 3 channels."
    assert images.shape[2] == Config.IMG_HEIGHT, "Height mismatch."
    assert images.shape[3] == Config.IMG_WIDTH, "Width mismatch."

    # Verify Masks are binary
    unique_vals = torch.unique(masks)
    assert (
        (unique_vals == 0) | (unique_vals == 1)
    ).all(), "Masks should be binary (0 or 1)."
    print("Data Loader verified.")

    # =========================================================================
    # 4. Model Initialization and Forward Pass Check
    # =========================================================================
    print("\n>>> Step 4: Model Initialization & Forward Pass")

    device = Config.DEVICE
    print(f"Using device: {device}")

    model = SaltUNetPlusPlus(deep_supervision=True).to(device)

    # Move batch to device
    images = images.to(device, dtype=torch.float32)
    masks = masks.to(device, dtype=torch.float32)

    # Test Phase 1 Forward (Deep Supervision = True)
    model.train()
    model.deep_supervision = True
    outputs_phase1 = model(images)

    assert isinstance(
        outputs_phase1, list
    ), "Phase 1 output should be a list (Deep Supervision)."
    assert (
        len(outputs_phase1) == 4
    ), "Phase 1 should return 4 outputs (3 aux + 1 final)."
    print("Phase 1 Forward Pass successful.")

    # Test Phase 2 Forward (Deep Supervision = False)
    model.deep_supervision = False
    outputs_phase2 = model(images)

    assert torch.is_tensor(outputs_phase2), "Phase 2 output should be a single tensor."
    assert outputs_phase2.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Output shape mismatch. Got {outputs_phase2.shape}"
    print("Phase 2 Forward Pass successful.")

    # =========================================================================
    # 5. Loss Calculation Check
    # =========================================================================
    print("\n>>> Step 5: Loss Function Verification")

    # Check BCEDiceLoss (Phase 1)
    criterion_p1 = BCEDiceLoss()
    loss_p1 = 0
    for logits in outputs_phase1:
        loss_p1 += criterion_p1(logits, masks)

    print(f"Phase 1 Loss: {loss_p1.item():.4f}")
    assert not torch.isnan(loss_p1), "Phase 1 Loss is NaN."

    # Check LovaszHingeLoss (Phase 2)
    criterion_p2 = LovaszHingeLoss(per_image=True)
    loss_p2 = criterion_p2(outputs_phase2, masks)

    print(f"Phase 2 Loss: {loss_p2.item():.4f}")
    assert not torch.isnan(loss_p2), "Phase 2 Loss is NaN."

    # =========================================================================
    # 6. Engine Training Loop Simulation
    # =========================================================================
    print("\n>>> Step 6: Running Engine Training Simulation")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PHASE1_LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    engine = SaltEngine(model, optimizer, device, scheduler)

    # --- Epoch 0: Phase 1 (Deep Supervision) ---
    print("--- Simulating Epoch 0 (Phase 1) ---")
    loss_epoch0 = engine.train_one_epoch(train_loader, epoch=0)
    assert loss_epoch0 > 0, "Training loss should be positive."

    # --- Validation ---
    print("--- Simulating Validation ---")
    val_map = engine.validate_one_epoch(val_loader)
    print(f"Validation mAP: {val_map:.4f}")

    # --- Epoch 21: Phase 2 (Metric Fine-tuning) ---
    # We simulate a later epoch to trigger the phase switch logic in train_one_epoch
    print("--- Simulating Epoch 21 (Phase 2) ---")
    # Note: We use the same loader for speed
    loss_epoch21 = engine.train_one_epoch(train_loader, epoch=21)
    assert loss_epoch21 > 0, "Training loss should be positive."

    # Save a dummy checkpoint
    torch.save(
        model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")
    )
    print("Checkpoint saved.")

    # =========================================================================
    # 7. Inference and Submission
    # =========================================================================
    print("\n>>> Step 7: Inference and Submission Generation")

    # Load Test Loader
    # Note: get_test_loader loads the full test set metadata (1000 samples).
    # Since we are in demo mode, we will just run prediction on it.
    # The inference is fast enough (1000 images ~1 min on GPU).
    test_loader = get_test_loader(load_cached_data=False)

    print(f"Test Loader batches: {len(test_loader)}")

    # Run Prediction (TTA enabled)
    print("Running prediction (this may take a moment)...")
    predictions, ids = engine.predict(test_loader, tta=True)

    assert len(predictions) == len(ids), "Mismatch between predictions and IDs."
    assert predictions[0].shape == (
        101,
        101,
    ), f"Prediction shape mismatch. Expected (101, 101), got {predictions[0].shape}"

    # Save Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    engine.save_predictions(predictions, ids, sub_path, threshold=0.5)

    # Verify Submission File
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) == 1000, f"Expected 1000 predictions, got {len(df_sub)}"

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
