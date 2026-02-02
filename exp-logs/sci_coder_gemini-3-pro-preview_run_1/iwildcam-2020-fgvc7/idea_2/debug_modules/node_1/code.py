import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import cv2

# Import from the provided library
from library import config
from library import utils
from library import box_utils
from library import dataset
from library import model
from library import loss
from library import engine


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n--- Setting up configuration for fast demonstration ---")
    utils.seed_everything(42)

    # Define temporary paths for mini-datasets
    temp_dir = "./working/demo_temp"
    os.makedirs(temp_dir, exist_ok=True)

    mini_train_path = os.path.join(temp_dir, "mini_train.csv")
    mini_val_path = os.path.join(temp_dir, "mini_val.csv")
    mini_test_path = os.path.join(temp_dir, "mini_test.csv")
    mini_submission_path = os.path.join(temp_dir, "mini_submission.csv")

    # Create mini-datasets by sampling the original metadata
    # We use a very small sample size to ensure the script runs quickly
    print("Creating mini-datasets...")
    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(config.VAL_METADATA_PATH)
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample 16 images for train, 8 for val, 8 for test
    # Ensure we pick images that actually exist (though metadata generator verified this)
    mini_train_df = full_train_df.sample(n=16, random_state=42)
    mini_val_df = full_val_df.sample(n=8, random_state=42)
    mini_test_df = full_test_df.sample(n=8, random_state=42)

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)
    mini_test_df.to_csv(mini_test_path, index=False)

    # Monkey-patch config to use these mini files and smaller parameters
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path
    config.SUBMISSION_FILE = mini_submission_path
    config.IMG_SIZE = 128  # Reduce size for speed
    config.BATCH_SIZE = 4  # Small batch size
    config.EPOCHS = 1  # Single epoch
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use CPU if GPU not available, though environment says A100 is available
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Verify Box Utilities
    print("\n--- Verifying Box Utilities ---")

    # Test loading megadetector data
    # Note: This might parse the large JSON if cache doesn't exist, which is expected behavior.
    bbox_df = box_utils.load_megadetector_data(load_cached_data=True)

    assert isinstance(
        bbox_df, pd.DataFrame
    ), "load_megadetector_data should return a DataFrame"
    expected_cols = ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "conf"]
    assert all(
        col in bbox_df.columns for col in expected_cols
    ), f"Missing columns in bbox df. Expected {expected_cols}"
    print("MegaDetector data loaded successfully.")

    # Test crop logic
    # Create a dummy bbox: center of a 100x100 image, size 20x20
    # Normalized: x=0.4, y=0.4, w=0.2, h=0.2
    dummy_bbox = [0.4, 0.4, 0.2, 0.2]
    img_w, img_h = 100, 100
    margin = 0.0  # No margin for exact calculation check

    x_min, y_min, x_max, y_max = box_utils.get_context_square_crop(
        dummy_bbox, img_w, img_h, margin=margin
    )

    # Center is (50, 50). Max side is 20. Half side is 10.
    # Expected: 40, 40, 60, 60
    assert x_min == 40 and y_min == 40, f"Crop coords mismatch. Got {x_min}, {y_min}"
    assert x_max == 60 and y_max == 60, f"Crop coords mismatch. Got {x_max}, {y_max}"
    print("Crop logic verified.")

    # 3. Verify Dataset and DataLoader
    print("\n--- Verifying Dataset and DataLoader ---")

    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # Fetch one batch from train loader
    images, targets = next(iter(train_loader))

    # Verify shapes
    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets shape: {targets.shape}")

    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Incorrect image tensor shape: {images.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Incorrect target tensor shape: {targets.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.long, "Targets should be long (int64)"
    print("DataLoaders verified.")

    # 4. Verify Model
    print("\n--- Verifying Model ---")

    # Initialize model
    net = model.get_model(device=device, pretrained=False)  # Pretrained=False for speed

    # Forward pass with dummy input
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    with torch.no_grad():
        outputs = net(dummy_input)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {config.NUM_CLASSES}), got {outputs.shape}"
    print("Model architecture verified.")

    # 5. Verify Loss Function
    print("\n--- Verifying Focal Loss ---")

    criterion = loss.FocalLoss()
    dummy_logits = torch.randn(4, config.NUM_CLASSES)
    dummy_targets = torch.randint(0, config.NUM_CLASSES, (4,))

    loss_val = criterion(dummy_logits, dummy_targets)

    print(f"Calculated Loss: {loss_val.item()}")
    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() > 0, "Loss should be positive"
    print("Loss function verified.")

    # 6. Verify Training Engine
    print("\n--- Verifying Training Engine ---")

    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Run training loop (1 epoch as per modified config)
    best_acc = engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config.EPOCHS,
        patience=1,
    )

    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    assert 0.0 <= best_acc <= 1.0, "Accuracy should be between 0 and 1"

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(config.WORKING_DIR, "checkpoint.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print("Training engine verified.")

    # 7. Verify Submission Generation
    print("\n--- Verifying Submission Generation ---")

    engine.generate_submission(
        net, test_loader, device, output_path=config.SUBMISSION_FILE
    )

    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    # Check submission content
    sub_df = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(sub_df) == len(
        mini_test_df
    ), f"Submission row count mismatch. Expected {len(mini_test_df)}, got {len(sub_df)}"

    print("Submission generation verified.")

    # 8. Cleanup
    print("\n--- Cleanup ---")
    shutil.rmtree(temp_dir)
    print("Temporary files removed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
