import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.models import CustomResNet, CustomDenseNet
from library.train import train_ensemble
from library.inference import generate_ensemble_predictions


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up configuration for fast demonstration...")

    # Override Config defaults to ensure the script runs quickly (Demo Mode)
    Config.DEBUG = True  # Use a small subset of data
    Config.DEBUG_SAMPLE_SIZE = 64  # Number of samples for debug
    Config.BATCH_SIZE = 16  # Small batch size
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.ARCHITECTURES = ["resnet"]  # Only train ResNet for this demo
    Config.SEEDS = [42]  # Single seed to save time

    # Redirect output directories to a demo folder
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    Config.setup_directories()

    # Set global seed for reproducibility
    set_seed(42)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ==========================================
    # 2. Dataset & DataLoader Demonstration
    # ==========================================
    print("\n>>> Verifying Data Loading...")

    # Initialize DataLoaders
    # We set num_workers=0 to avoid overhead in this short script
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        load_cached_data=False,  # Force processing from metadata
        debug=Config.DEBUG,
    )

    # Fetch a single batch from the training loader
    images, labels = next(iter(train_loader))

    # Validate shapes
    print(f"    Train Batch Images Shape: {images.shape}")
    print(f"    Train Batch Labels Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, 32, 32)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"

    # Validate data range (ToTensor scales to [0, 1])
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized between 0 and 1."

    print("    Data Loading verification passed.")

    # ==========================================
    # 3. Model Architecture Demonstration
    # ==========================================
    print("\n>>> Verifying Model Architectures...")

    # Instantiate models
    resnet_model = CustomResNet(num_classes=1)
    densenet_model = CustomDenseNet(num_classes=1)

    # Move to CPU for simple shape verification
    resnet_model.cpu()
    densenet_model.cpu()

    # Perform dummy forward pass
    with torch.no_grad():
        out_res = resnet_model(images.cpu())
        out_dense = densenet_model(images.cpu())

    print(f"    ResNet Output Shape: {out_res.shape}")
    print(f"    DenseNet Output Shape: {out_dense.shape}")

    # Assert output shapes are (Batch_Size, Num_Classes)
    assert out_res.shape == (Config.BATCH_SIZE, 1), "ResNet output shape mismatch"
    assert out_dense.shape == (Config.BATCH_SIZE, 1), "DenseNet output shape mismatch"

    print("    Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n>>> Running Training Ensemble (ResNet, 1 Epoch, Seed 42)...")

    # Run the training routine provided in library.train
    # This will train based on Config.ARCHITECTURES and Config.SEEDS
    train_ensemble()

    # Verify that the model checkpoint was saved
    expected_model_path = Config.get_model_path("resnet", 42)
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"

    print(f"    Training complete. Model saved to: {expected_model_path}")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\n>>> Running Inference Generation...")

    # Run the inference routine provided in library.inference
    # This loads the saved models, predicts on the test set, and saves submission.csv
    generate_ensemble_predictions()

    # Verify submission file existence
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission file content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file loaded. Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    assert (
        "id" in sub_df.columns and "has_cactus" in sub_df.columns
    ), "Submission file missing required columns."

    # Check that predictions are probabilities
    preds = sub_df["has_cactus"].values
    assert np.all(preds >= 0.0) and np.all(
        preds <= 1.0
    ), "Predictions must be probabilities between 0 and 1."

    print("    Inference verification passed.")
    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    main()
