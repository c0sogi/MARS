import sys
import os
import numpy as np
import torch
import pandas as pd
from unittest.mock import MagicMock

# --- 1. Environment Patching ---
# The provided library/dataset.py imports pydicom.
# If pydicom is missing in the environment, we mock it to allow the library to load.
try:
    import pydicom
except ImportError:
    # Create a mock for pydicom that returns a dummy image
    mock_pydicom = MagicMock()

    def mock_dcmread(path, stop_before_pixels=False):
        dataset = MagicMock()
        # Return a random image to simulate data (1024x1024)
        # Using a deterministic pattern based on path hash could be better, but random is sufficient for demo
        dataset.pixel_array = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows = 1024
        dataset.Columns = 1024
        return dataset

    mock_pydicom.dcmread = mock_dcmread
    sys.modules["pydicom"] = mock_pydicom

# --- 2. Imports from Library ---
# These imports must happen after the mock
from library.config import Config, seed_everything
from library.dataset import VinDrDataset
from library.model import get_model
from library.engine import fit, inference
from library.utils import collate_fn


def main():
    print("Initializing Thoracic Lung Disease Detection Pipeline Demo...")

    # --- Configuration Overrides ---
    # Modify Config class attributes to optimize for speed and demonstration purposes
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce computational load
    Config.IMAGE_SIZE = 512  # Resize to 512x512 for faster training
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch for demonstration
    Config.DATASET_FRACTION = 0.02  # Use 2% of the data
    Config.NUM_WORKERS = 2  # Adjust workers

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- Step 1: Dataset & DataLoader ---
    print("\n[1/4] Setting up Datasets...")

    # Initialize Training Dataset
    train_dataset = VinDrDataset(mode="train", dataset_fraction=Config.DATASET_FRACTION)
    print(f"Train Dataset Size: {len(train_dataset)}")

    # Verify Dataset Item Structure
    if len(train_dataset) > 0:
        img, target = train_dataset[0]
        assert isinstance(img, torch.Tensor), "Image must be a torch.Tensor"
        assert img.shape == (
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Image shape mismatch. Got {img.shape}"
        assert "boxes" in target, "Target dict missing 'boxes'"
        assert "labels" in target, "Target dict missing 'labels'"
        # Check box coordinates
        if len(target["boxes"]) > 0:
            assert target["boxes"].shape[1] == 4, "Boxes must be (N, 4)"

    # Initialize Validation Dataset
    val_dataset = VinDrDataset(mode="val", dataset_fraction=Config.DATASET_FRACTION)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Verify DataLoader Batch
    batch_imgs, batch_targets = next(iter(train_loader))
    assert len(batch_imgs) == len(
        batch_targets
    ), "Batch size mismatch between images and targets"

    # --- Step 2: Model Initialization ---
    print("\n[2/4] Initializing Faster R-CNN Model...")
    model = get_model(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Verify Model Output Format (Dummy Forward Pass)
    # We use the batch fetched above. Model needs to be in train mode to return loss dict.
    model.train()
    # Move batch to device
    batch_imgs_dev = [img.to(device) for img in batch_imgs]
    batch_targets_dev = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in batch_targets
    ]

    loss_dict = model(batch_imgs_dev, batch_targets_dev)
    assert isinstance(
        loss_dict, dict
    ), "Model forward pass in train mode should return a dict"
    assert "loss_classifier" in loss_dict, "Loss dict missing classification loss"
    assert "loss_box_reg" in loss_dict, "Loss dict missing regression loss"

    # --- Step 3: Training Loop ---
    print("\n[3/4] Starting Training (1 Epoch)...")

    # Setup Optimizer and Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )

    # Execute Training
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("Training finished and model saved.")

    # --- Step 4: Inference ---
    print("\n[4/4] Generating Predictions on Test Set...")

    # Initialize Test Dataset (using a small fraction for demo speed)
    test_dataset = VinDrDataset(mode="test", dataset_fraction=0.02)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Run Inference
    df_submission = inference(model, test_loader, device)

    # Verify Submission DataFrame
    assert isinstance(df_submission, pd.DataFrame), "Inference returned invalid type"
    assert not df_submission.empty, "Submission DataFrame is empty"
    assert "image_id" in df_submission.columns, "Submission missing 'image_id' column"
    assert (
        "PredictionString" in df_submission.columns
    ), "Submission missing 'PredictionString' column"

    # Verify Prediction String Format
    sample_pred = df_submission.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str), "PredictionString is not a string"
    print(f"Sample Prediction: {sample_pred}")

    # Save submission (optional, but good for verification)
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    df_submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
