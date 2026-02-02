import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode
from library.dataset import get_loaders
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss
from library.engine import SaltEngine
from library.inference import predict_with_tta, optimize_threshold, generate_submission

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuration & Setup ---")

    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Re-run setup to create the new directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated for demo run.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n--- 2. Data Loading Verification ---")

    # Load debug subset (100 train images, 50 val images)
    # load_cached_data=False ensures we test the raw image processing logic
    train_loader, val_loader = get_loaders(fold=0, load_cached_data=False, debug=True)

    # Fetch a single batch
    images, masks = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Masks Shape:  {masks.shape}")

    # Assertions
    # 1. Check Batch Size
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}"

    # 2. Check Channels (3 channels: Seismic, Seismic, Depth)
    assert images.shape[1] == 3, "Expected 3 input channels"

    # 3. Check Spatial Dimensions (Padded to 128x128)
    assert images.shape[2:] == (128, 128), "Expected images padded to 128x128"
    assert masks.shape[2:] == (128, 128), "Expected masks padded to 128x128"

    # 4. Check Value Ranges
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"
    assert set(np.unique(masks.numpy())).issubset(
        {0, 1}
    ), "Masks should be binary (0, 1)"

    print("Data loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n--- 3. Model Architecture Verification ---")

    # Initialize model with random weights (encoder_weights=None) for speed/offline safety
    model = SaltUNetPlusPlus(encoder_weights=None)
    model.to(Config.DEVICE)

    # Test Forward Pass (Training Mode)
    model.train()
    images_gpu = images.to(Config.DEVICE)
    outputs = model(images_gpu)

    # Deep Supervision check: Should return a list of tensors
    print(f"Model Output Type: {type(outputs)}")
    print(f"Number of Output Heads: {len(outputs)}")

    assert isinstance(
        outputs, list
    ), "Model in train mode should return a list (Deep Supervision)"
    assert len(outputs) == 4, "Expected 4 output heads from U-Net++"
    assert outputs[-1].shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Output shape mismatch"

    print("Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n--- 4. Loss Function Verification ---")

    criterion = BCEDiceLoss()
    masks_gpu = masks.to(Config.DEVICE)

    # Calculate loss on the final head
    loss = criterion(outputs[-1], masks_gpu)

    print(f"Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Loss function verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n--- 5. Training Loop Execution ---")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    engine = SaltEngine(model, Config.DEVICE, optimizer)

    print("Starting training for 1 epoch...")
    save_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")

    # Run fit (train + validate)
    engine.fit(train_loader, val_loader, epochs=1, save_path=save_path, patience=1)

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Inference & Utils Verification
    # -------------------------------------------------------------------------
    print("\n--- 6. Inference & Utils Verification ---")

    # Run Inference (TTA + Cropping)
    print("Running inference on validation set...")
    results = predict_with_tta(model, val_loader, Config.DEVICE)

    preds = results["preds"]
    targets = results["targets"]

    print(f"Predictions Shape: {preds.shape}")

    # Verify Automatic Cropping (128x128 -> 101x101)
    assert preds.shape[1:] == (
        101,
        101,
    ), "Predictions should be cropped to original size (101x101)"
    assert targets.shape[1:] == (
        101,
        101,
    ), "Targets should be cropped to original size (101x101)"

    # Optimize Threshold
    best_thresh = optimize_threshold(preds, targets)

    # Verify RLE Encoding/Decoding
    print("Verifying RLE logic...")
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[50:60, 50:60] = 1  # Create a 10x10 square

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded)

    assert np.array_equal(dummy_mask, decoded), "RLE Round-trip failed"
    print("RLE logic verified.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- 7. Submission Generation ---")

    # Get IDs corresponding to the validation set
    # Note: val_loader yields (images, masks), so we extract IDs from the dataset directly
    val_ids = val_loader.dataset.ids

    # Ensure lengths match
    assert len(val_ids) == len(preds), "Mismatch between IDs and Predictions count"

    generate_submission(preds, val_ids, best_thresh, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(df_sub)}")
    print(df_sub.head())

    assert len(df_sub) == len(val_ids), "Submission row count mismatch"
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing"

    print("\n=== Demo Execution Completed Successfully ===")
