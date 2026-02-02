import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders, get_test_dataloader
from library.model import get_species_classifier
from library.engine import train_model, generate_submission


def run_demonstration():
    print("=== Starting Demonstration ===")

    # 1. Setup and Configuration
    # We override some Config parameters to ensure the demo runs quickly
    print("Setting up configuration for rapid demonstration...")
    Config.setup()
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Smaller batch size for the small subset
    Config.DEBUG = True

    # Set random seed for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    # We limit the data to 200 samples to verify the pipeline runs without waiting for hours
    print("\n=== Loading Data (Subset) ===")
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduced workers for small script
        limit_data=200,  # Limit to 200 images for speed
    )

    # Validation: Check Data Shapes
    print("Verifying DataLoader shapes...")
    images, labels = next(iter(train_loader))

    # Expected shape: [Batch_Size, 3, Height, Width]
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        images.shape == expected_img_shape
    ), f"Image batch shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Expected shape: [Batch_Size]
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label batch shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = get_species_classifier(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,  # Freeze backbone for faster demo training
    )
    model = model.to(device)

    # Validation: Check Model Output Shape
    print("Verifying model output shape...")
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: [2, NUM_CLASSES]
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model output shape verified.")

    # 4. Training Loop
    print("\n=== Starting Training Loop ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training (Config.EPOCHS is set to 1)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.EPOCHS,
        save_path=os.path.join(Config.WORKING_DIR, "demo_model.pth"),
    )

    # 5. Inference and Submission
    print("\n=== Generating Submission ===")

    # Load Test Data
    # Note: We don't limit test data here to ensure we generate a full valid submission file,
    # but since inference is fast and we just want to prove it works, we will proceed.
    # The test loader only loads high-confidence images.
    test_loader, low_conf_df = get_test_dataloader(
        batch_size=Config.BATCH_SIZE * 2, num_workers=2  # Larger batch for inference
    )

    # Generate Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    generate_submission(
        model=model,
        test_loader=test_loader,
        low_conf_df=low_conf_df,
        device=device,
        output_path=submission_path,
    )

    # 6. Verify Submission File
    print("\n=== Verifying Submission File ===")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Check columns
    assert "Id" in df_sub.columns, "Missing 'Id' column in submission."
    assert "Predicted" in df_sub.columns, "Missing 'Predicted' column in submission."

    # Check total rows matches test metadata
    # We read the test metadata directly to compare counts
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) does not match test metadata ({len(df_test_meta)})."

    # Check for NaN values
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Submission file verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
