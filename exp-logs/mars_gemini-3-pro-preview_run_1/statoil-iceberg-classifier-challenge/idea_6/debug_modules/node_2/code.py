import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataset
from library.model import IcebergResNet
from library.train import run_fold


def main():
    # 1. Setup and Configuration Override
    print("--- Setting up Configuration ---")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration run
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Dataset Demonstration
    print("\n--- Verifying Dataset Loading ---")

    # Load train dataset
    train_ds = get_dataset("train", load_cached_data=False)

    # Assertions to verify dataset size matches debug config
    expected_len = min(Config.DEBUG_SAMPLE_SIZE, 1026)  # 1026 is total train size
    assert (
        len(train_ds) == expected_len
    ), f"Expected {expected_len} samples, got {len(train_ds)}"

    # Fetch a single sample to verify structure
    img, angle, label, sample_id = train_ds[0]

    # Verify shapes
    # Image should be (3, 224, 224) due to Albumentations ToTensorV2 and resizing
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    # Angle should be a scalar tensor
    assert isinstance(angle, torch.Tensor), "Angle is not a tensor"
    # Label should be a scalar tensor
    assert isinstance(label, torch.Tensor), "Label is not a tensor"
    # ID should be a string
    assert isinstance(
        sample_id, (str, np.str_)
    ), f"ID is not a string: {type(sample_id)}"

    print("Dataset verification passed.")
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Angle: {angle.item()}")
    print(f"Sample Label: {label.item()}")

    # 3. Model Demonstration
    print("\n--- Verifying Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IcebergResNet().to(device)

    # Create dummy input batch
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_ang = torch.tensor([35.0, 40.0]).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    # Verify output shape (Batch_Size, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model forward pass successful.")

    # 4. Training Demonstration
    print("\n--- Running Training Loop (Fold 0) ---")

    # Run training using the library function
    # This handles dataloaders, optimizer, loop, and saving best model
    trained_model = run_fold(fold_idx=0)

    # Verify model file was created
    model_path = os.path.join(Config.WORKING_DIR, "model_fold_0_best.pth")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"

    print("Training simulation complete.")

    # 5. Inference and Submission
    print("\n--- Running Inference on Test Set ---")

    # Load test dataset
    test_ds = get_dataset("test", load_cached_data=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    trained_model.eval()
    predictions = []
    ids = []

    print(f"Predicting on {len(test_ds)} test samples...")

    with torch.no_grad():
        for images, angles, sample_ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = trained_model(images, angles)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids.extend(sample_ids)

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Verify submission content
    assert len(df_sub) == len(
        test_ds
    ), "Mismatch between prediction count and test set size"
    assert df_sub["is_iceberg"].min() >= 0.0, "Probabilities should be >= 0"
    assert df_sub["is_iceberg"].max() <= 1.0, "Probabilities should be <= 1"

    # Save submission
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print first few rows
    print("\nSubmission Head:")
    print(df_sub.head())

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
