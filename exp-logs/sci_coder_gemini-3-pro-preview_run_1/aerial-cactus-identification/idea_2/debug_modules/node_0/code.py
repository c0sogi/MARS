import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import CactusDataset, get_transforms
from library.model import CactusDenseNet
from library.engine import (
    train_one_epoch,
    validate,
    predict_with_tta,
    generate_submission,
)

if __name__ == "__main__":
    # 1. Setup & Configuration
    print("Setting up configuration and environment...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images for training/val demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Pipeline Verification
    print("Initializing datasets and dataloaders...")

    # Initialize Datasets (Debug mode = small subset)
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        debug=True,
    )
    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_transforms("val"),
        debug=True,
    )

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Assertion: Check batch structure
    images, labels = next(iter(train_loader))

    # Expected shape: (Batch, Channels, Height, Width) -> (32, 3, 32, 32)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Incorrect image batch shape: {images.shape}"

    # Expected label shape: (Batch,) -> (32,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label batch shape: {labels.shape}"

    print("Data pipeline verified successfully.")

    # 3. Model Instantiation & Forward Pass
    print("Initializing model...")
    model = CactusDenseNet(num_classes=1)
    model.to(device)

    # Assertion: Check forward pass dimensions
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Output should be (Batch, Num_Classes) -> (2, 1)
    assert output.shape == (2, 1), f"Incorrect model output shape: {output.shape}"
    print("Model initialized and verified successfully.")

    # 4. Training & Validation Loop
    print("Starting training loop demonstration...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run for defined epochs (1 in this demo)
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Assertion: Loss should be a float
        assert isinstance(train_loss, float), "Train loss is not a float"

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Assertion: Metrics should be floats
        assert isinstance(val_loss, float), "Val loss is not a float"
        assert isinstance(val_auc, float), "Val AUC is not a float"

    print("Training loop completed successfully.")

    # 5. Inference & Submission
    print("Starting inference and submission generation...")

    # For submission, we must use the full test set because generate_submission
    # checks predictions against the full test metadata file.
    # The test set is small (~3k images), so this is fast.
    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        transform=get_transforms("test"),
        debug=False,  # Must use full set
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch size for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Generate submission
    # This function uses predict_with_tta internally and saves to CSV
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Assertion: Check if file was created
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Assertion: Check file content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission file missing required columns"
    assert len(df_sub) == len(
        test_dataset
    ), f"Submission row count mismatch. Expected {len(test_dataset)}, got {len(df_sub)}"

    print(f"Submission verified. File saved at: {Config.SUBMISSION_PATH}")
    print("Demonstration script finished successfully.")
