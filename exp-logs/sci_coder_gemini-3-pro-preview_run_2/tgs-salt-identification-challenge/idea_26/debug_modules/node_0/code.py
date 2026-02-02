import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import random
import cv2
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    calc_iou,
    calc_map_score,
)
from library.dataset import SaltDataset
from library.model import ResNet34WideLinkNet
from library.losses import MultiTaskLoss
from library.engine import train_one_epoch, validate, train_model, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting Salt Segmentation Pipeline Demonstration...")

    # 1. Setup and Configuration
    set_seed(Config.SEED)
    Config.set_debug_mode(
        True
    )  # Enable debug mode for speed (fewer epochs, smaller dataset)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")

    # 2. Verify Utilities
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded_mask), "RLE Decode mismatch!"
    print("RLE Encode/Decode: Passed")

    # Test Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded_img = pad_image(dummy_img)
    assert padded_img.shape == (128, 128), f"Padding shape mismatch: {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, orig_shape=(101, 101))
    # Note: Reflection padding might not perfectly reconstruct edges if not careful,
    # but unpad simply crops. For the center region, it should match if logic is correct.
    # Here we just check shapes.
    assert unpadded_img.shape == (
        101,
        101,
    ), f"Unpadding shape mismatch: {unpadded_img.shape}"
    print("Image Padding/Unpadding: Passed")

    # Test Metrics
    iou = calc_iou(dummy_mask, dummy_mask)
    assert iou == 1.0, "IoU calculation failed for identical masks"
    map_score = calc_map_score(dummy_mask, dummy_mask)
    assert map_score == 1.0, "mAP calculation failed for identical masks"
    print("Metric Calculation: Passed")

    # 3. Dataset and DataLoader
    print("\n--- Initializing Datasets ---")

    # Initialize Datasets
    # Note: This will create cache files in working/idea_26/cache
    train_dataset = SaltDataset(mode="train", load_cached_data=False)
    val_dataset = SaltDataset(mode="val", load_cached_data=False)

    print(f"Train Dataset Size (Debug): {len(train_dataset)}")
    print(f"Val Dataset Size (Debug): {len(val_dataset)}")

    # Verify item structure
    sample = train_dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert "depth" in sample
    assert "id" in sample

    # Check shapes
    # Image: (1, 128, 128) -> 1 channel due to Config.IN_CHANNELS=1 logic in dataset
    assert sample["image"].shape == (
        1,
        128,
        128,
    ), f"Image tensor shape wrong: {sample['image'].shape}"
    assert sample["mask"].shape == (
        1,
        128,
        128,
    ), f"Mask tensor shape wrong: {sample['mask'].shape}"
    assert sample["depth"].shape == (
        1,
    ), f"Depth tensor shape wrong: {sample['depth'].shape}"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script safety
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    print("DataLoaders initialized.")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = ResNet34WideLinkNet()
    model.to(Config.DEVICE)

    # Test Forward Pass
    dummy_input = torch.randn(2, 1, 128, 128).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert "mask" in output and "depth" in output
    assert output["mask"].shape == (
        2,
        1,
        128,
        128,
    ), f"Model mask output shape mismatch: {output['mask'].shape}"
    assert output["depth"].shape == (
        2,
        1,
    ), f"Model depth output shape mismatch: {output['depth'].shape}"
    print("Model forward pass successful.")

    # 5. Loss Function
    print("\n--- Testing Loss Function ---")
    criterion = MultiTaskLoss()

    # Create dummy targets
    dummy_target_mask = torch.randint(0, 2, (2, 1, 128, 128)).float().to(Config.DEVICE)
    dummy_target_depth = torch.randn(2, 1).float().to(Config.DEVICE)
    targets = {"mask": dummy_target_mask, "depth": dummy_target_depth}

    loss, loss_dict = criterion(output, targets)
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")

    # 6. Training Loop Simulation
    print("\n--- Running Training Loop (Simulation) ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train one epoch
    epoch_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE, epoch=1)
    assert epoch_loss > 0, "Training loss should be positive"

    # Validate
    val_loss, val_map = validate(model, val_loader, Config.DEVICE)
    print(f"Validation complete. Loss: {val_loss:.4f}, mAP: {val_map:.4f}")

    # 7. Full Training Orchestration
    print("\n--- Running Full Training Orchestration ---")
    save_path = os.path.join(Config.CHECKPOINTS_DIR, "demo_model.pth")

    # Using the train_model function from engine.py
    # This handles the loop, early stopping, and threshold optimization
    best_threshold = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        Config.DEVICE,
        epochs=2,  # Small number for demo
        patience=1,
        save_path=save_path,
    )

    print(f"Optimal Threshold found: {best_threshold}")
    assert os.path.exists(save_path), "Model checkpoint was not saved."

    # 8. Inference and Submission
    print("\n--- Running Inference on Test Set ---")
    test_dataset = SaltDataset(mode="test", load_cached_data=False)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Load the best model state
    model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))

    predict_and_submit(
        model,
        test_loader,
        Config.DEVICE,
        threshold=best_threshold,
        submission_path=submission_path,
    )

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify submission format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")
    assert "id" in sub_df.columns and "rle_mask" in sub_df.columns

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
