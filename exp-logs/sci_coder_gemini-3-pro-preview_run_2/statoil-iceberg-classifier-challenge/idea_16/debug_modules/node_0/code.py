import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data_utils import load_data, IcebergDataset, set_seed
from library.model_utils import CSNet
from library.train_utils import train_fold


def run_demo():
    print("Initializing Demo...")

    # 1. Configure for Speed/Debug
    # We modify the global Config to run a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.NUM_EPOCHS = 2  # Minimal epochs
    Config.BATCH_SIZE = 8
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.MODEL_DIR = Config.WORK_DIR
    Config.SUBMISSION_DIR = Config.WORK_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Re-run setup to ensure new directories exist
    Config.setup_directories()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Load Data
    print("Loading data...")
    # load_data handles caching and splitting based on Config
    train_data, val_data, test_data = load_data(Config, load_cached_data=False)

    # Verification: Check data shapes
    print(f"Train images shape: {train_data['images'].shape}")
    assert (
        train_data["images"].shape[1] == 3
    ), "Expected 3 initial channels (HH, HV, Avg)"
    assert (
        len(train_data["images"]) == Config.DEBUG_SUBSET_SIZE
    ), "Debug subset size mismatch"

    # 3. Prepare Datasets and Loaders
    print("Preparing Datasets...")
    train_dataset = IcebergDataset(
        train_data["images"], train_data["angles"], train_data["labels"], transform=True
    )
    val_dataset = IcebergDataset(
        val_data["images"], val_data["angles"], val_data["labels"], transform=False
    )
    test_dataset = IcebergDataset(
        test_data["images"], test_data["angles"], labels=None, transform=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debug execution
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verification: Check Dataset Output (6 channels)
    # IcebergDataset augments 3 channels to 6 (Original + Inverted)
    sample_img, sample_angle, sample_label = train_dataset[0]
    assert sample_img.shape == (
        6,
        75,
        75,
    ), f"Expected (6, 75, 75), got {sample_img.shape}"
    print("Dataset verification passed: Correctly producing 6-channel tensors.")

    # 4. Model Initialization
    print("Initializing CSNet Model...")
    device = Config.DEVICE
    model = CSNet().to(device)

    # Verification: Dummy Forward Pass
    dummy_img = torch.randn(2, 6, 75, 75).to(device)
    dummy_angle = torch.randn(2, 1).to(device)
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model verification passed: Forward pass successful.")

    # 5. Training Loop Demonstration
    print("Starting Training Loop (Fold 0)...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Train for one fold
    trained_model, best_val_loss = train_fold(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        Config,
        fold_idx=0,
    )

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")

    # Verification: Checkpoint existence
    checkpoint_path = os.path.join(Config.MODEL_DIR, "csnet_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."
    print(f"Checkpoint verified at {checkpoint_path}")

    # 6. Inference and Submission
    print("Generating predictions on Test set...")
    trained_model.eval()
    predictions = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = trained_model(images, angles)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy().flatten())

    # Create Submission DataFrame
    # Note: In debug mode, test_data['ids'] is also subsetted
    submission_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": predictions})

    # Verify submission length matches test data subset
    assert len(submission_df) == len(test_data["ids"]), "Submission length mismatch"

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Verify File Content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(saved_df.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns incorrect"
    assert not saved_df.isnull().values.any(), "Submission contains NaNs"
    print("Submission file verification passed.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
