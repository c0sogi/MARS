import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.data import MRIDataset, get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.inference import predict_submission


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    print("1. Configuring environment for demo...")

    # Define a specific working directory for this demo to avoid overwriting existing work
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce compute load for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Model Path: {Config.MODEL_SAVE_PATH}")
    print("-" * 40)

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("2. Verifying Data Pipeline...")

    # Load metadata (assuming metadata generation script has already run as per instructions)
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    train_df = pd.read_csv(Config.TRAIN_METADATA)

    # Instantiate Dataset (Train Mode)
    # We use a small subset to speed up the ROI cache computation in the demo
    subset_df = train_df.head(10).copy()
    dataset = MRIDataset(subset_df, phase="train", load_cached_data=False)

    # Fetch a single item
    image, label = dataset[0]

    print(f"   Input Tensor Shape: {image.shape}")
    print(f"   Label: {label}")

    # Assertions
    # Shape should be (Channels, Height, Width) -> (12, 224, 224)
    # 12 channels = 4 modalities * 3 slices
    assert image.shape == (
        12,
        224,
        224,
    ), f"Expected shape (12, 224, 224), got {image.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    # Test DataLoader
    loader = get_dataloader(subset_df, phase="train", batch_size=Config.BATCH_SIZE)
    batch_images, batch_labels = next(iter(loader))

    assert batch_images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_images.shape[1] == 12, "Channel dimension mismatch in batch"

    print("   Data Pipeline verified successfully.")
    print("-" * 40)

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("3. Verifying Model Architecture...")

    device = Config.DEVICE
    model = AsymmetricEfficientNet(num_classes=1)
    model.to(device)
    model.eval()

    # Create a dummy input batch: (Batch_Size, Channels, H, W)
    dummy_input = torch.randn(2, 12, 224, 224).to(device)

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"   Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (2, 1), f"Expected logits shape (2, 1), got {logits.shape}"

    print("   Model Architecture verified successfully.")
    print("-" * 40)

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("4. Executing Training Loop (Debug Mode)...")

    # run_training with debug=True uses a small subset of data
    best_auc = run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Verify model file creation
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Training failed to save model at {Config.MODEL_SAVE_PATH}"
        )

    print(f"   Training complete. Best AUC: {best_auc:.4f}")
    print("   Model file exists.")
    print("-" * 40)

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Verification
    # --------------------------------------------------------------------------
    print("5. Verifying Inference and Submission...")

    # Run inference using the model we just trained
    predict_submission(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_FILE,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Inference failed to create submission file at {Config.SUBMISSION_FILE}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"   Submission Shape: {sub_df.shape}")
    print(f"   Columns: {sub_df.columns.tolist()}")

    # Assertions
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(sub_df.columns)}"

    # Check if probabilities are valid
    if not sub_df.empty:
        probs = sub_df["MGMT_value"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Predictions contain values outside [0, 1]"

    print("   Inference verified successfully.")
    print("-" * 40)

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
