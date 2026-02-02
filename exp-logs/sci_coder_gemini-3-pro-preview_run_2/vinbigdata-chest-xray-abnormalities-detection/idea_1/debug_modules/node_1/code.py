import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders, LungDataset
from library.model import get_model
from library.engine import train_model, predict_and_submit


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed/Demo
    # ---------------------------------------------------------
    print(">>> Step 1: Configuring environment...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config parameters to ensure quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Use modest workers for demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create working directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {get_device()}")

    # ---------------------------------------------------------
    # 2. Verify Data Loading Pipeline
    # ---------------------------------------------------------
    print("\n>>> Step 2: Verifying Data Loading...")

    # Use debug=True to load a tiny subset of the data
    train_loader, val_loader = get_dataloaders(debug=True)

    # Fetch one batch to verify structure and shapes
    images, targets, image_ids = next(iter(train_loader))

    # Assertions for batch structure
    assert isinstance(images, tuple) or isinstance(
        images, list
    ), "Images should be a list/tuple"
    assert isinstance(targets, tuple) or isinstance(
        targets, list
    ), "Targets should be a list/tuple"
    assert (
        len(images) == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {len(images)}"
    assert len(targets) == Config.BATCH_SIZE, "Number of targets must match batch size"

    # Assertions for image tensor
    # Shape: (C, H, W) -> (3, 640, 640)
    img_tensor = images[0]
    assert isinstance(img_tensor, torch.Tensor), "Image must be a torch Tensor"
    assert img_tensor.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img_tensor.shape}"
    assert img_tensor.dtype == torch.float32, "Image dtype should be float32"

    # Assertions for target dictionary
    target_dict = targets[0]
    expected_keys = {"boxes", "labels", "area", "iscrowd", "image_id"}
    assert expected_keys.issubset(
        target_dict.keys()
    ), f"Target missing keys. Found: {target_dict.keys()}"

    # Check bounding box shape if boxes exist
    if target_dict["boxes"].shape[0] > 0:
        assert (
            target_dict["boxes"].shape[1] == 4
        ), "Bounding boxes must have 4 coordinates"
        assert (
            target_dict["labels"].shape[0] == target_dict["boxes"].shape[0]
        ), "Labels count must match boxes count"

    print("Data Loading verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Model Instantiation and Forward Pass
    # ---------------------------------------------------------
    print("\n>>> Step 3: Verifying Model...")

    # Initialize model (pretrained=False for speed in demo to avoid large downloads if not cached)
    # In a real run, keep pretrained=True.
    model = get_model(pretrained=False)
    device = get_device()
    model.to(device)

    # Move batch to device
    images_dev = [img.to(device) for img in images]
    targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Set to train mode to get loss dictionary
    model.train()
    loss_dict = model(images_dev, targets_dev)

    # Verify output
    assert isinstance(
        loss_dict, dict
    ), "Model output in train mode should be a dictionary"
    assert "classification" in loss_dict, "Loss dict missing classification loss"
    assert "bbox_regression" in loss_dict, "Loss dict missing bbox regression loss"

    # Calculate total loss
    total_loss = sum(loss for loss in loss_dict.values())
    assert not torch.isnan(total_loss), "Loss should not be NaN"

    print(
        f"Model Forward Pass verification passed. Initial Loss: {total_loss.item():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Verify Training Engine
    # ---------------------------------------------------------
    print("\n>>> Step 4: Running Training Loop (Debug Mode)...")

    # train_model(debug=True) runs the full training/validation loop on the subset
    # and saves the best model to Config.WORKING_DIR/best_model.pth
    best_model_path = train_model(debug=True)

    assert os.path.exists(best_model_path), f"Best model not found at {best_model_path}"
    print(f"Training loop completed. Model saved to {best_model_path}")

    # ---------------------------------------------------------
    # 5. Verify Inference and Submission
    # ---------------------------------------------------------
    print("\n>>> Step 5: Verifying Inference...")

    # Create a small subset of test metadata to speed up inference verification
    # We read the actual test metadata, take top 5 rows, and save to a temp file.
    original_test_meta_path = Config.TEST_META_PATH
    temp_test_meta_path = os.path.join(Config.WORKING_DIR, "test_meta_small.csv")

    df_test = pd.read_csv(original_test_meta_path)
    df_test_small = df_test.head(5).copy()
    df_test_small.to_csv(temp_test_meta_path, index=False)

    # Monkey-patch the Config to point to the small test set
    Config.TEST_META_PATH = temp_test_meta_path

    # Run inference using the trained model
    predict_and_submit(model_path=best_model_path)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == 5, f"Submission should have 5 rows, got {len(df_sub)}"
    assert "image_id" in df_sub.columns, "Submission missing image_id column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing PredictionString column"

    # Check format of PredictionString
    # Should be "class score xmin ymin xmax ymax ..."
    sample_pred = df_sub.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str), "PredictionString must be a string"
    parts = sample_pred.split()
    # Length should be multiple of 6 (class, score, x, y, x, y)
    assert (
        len(parts) % 6 == 0
    ), "PredictionString format invalid (length not multiple of 6)"

    print(
        f"Inference verification passed. Submission saved to {Config.SUBMISSION_PATH}"
    )
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
