import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, apk, mapk, get_transforms
from library.dataset import get_dataloaders, get_label_encoder
from library.model import HotelRecognitionModel
from library.train import run_training
from library.inference import predict_and_submit


def main():
    print("=== Starting Demonstration of Hotel Recognition Pipeline ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and resource efficiency
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demo
    Config.IMG_SIZE = 128  # Reduce resolution for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.BACKBONE = "resnet18"  # Lighter backbone than convnext_tiny
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Backbone: {Config.BACKBONE}")
    print(f"    Image Size: {Config.IMG_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Verify Utilities
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Verify AP@K logic
    # Case 1: Perfect prediction
    score_perfect = apk([10], [10, 20, 30, 40, 50], k=5)
    assert score_perfect == 1.0, f"Expected AP@5 of 1.0, got {score_perfect}"

    # Case 2: Correct item at 2nd position
    score_second = apk([10], [20, 10, 30, 40, 50], k=5)
    assert score_second == 0.5, f"Expected AP@5 of 0.5, got {score_second}"

    # Case 3: No correct item
    score_none = apk([10], [20, 30, 40, 50, 60], k=5)
    assert score_none == 0.0, f"Expected AP@5 of 0.0, got {score_none}"

    print("    AP@K metric logic verified.")

    # --------------------------------------------------------------------------
    # 3. Data Loading
    # --------------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")

    # Get dataloaders in debug mode
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        debug=Config.DEBUG,
        load_cached_data=False,  # Force re-computation for demo safety
    )

    print(f"    Number of classes: {len(classes)}")
    print(f"    Train batches: {len(train_loader)}")

    # Verify Train Batch Structure
    images, labels = next(iter(train_loader))

    # Assert Image Shape: (B, 3, H, W)
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"

    # Assert Label Shape: (B,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    # Verify Test Batch Structure (No labels)
    test_images = next(iter(test_loader))
    assert test_images.shape[1:] == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Test image shape mismatch."

    print("    DataLoaders verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[4] Initializing Model and Testing Forward Pass...")

    device = torch.device("cpu")  # Use CPU for simple shape verification
    model = HotelRecognitionModel(
        n_classes=len(classes),
        model_name=Config.BACKBONE,
        pretrained=False,  # Faster init
        embedding_size=Config.EMBEDDING_SIZE,
    ).to(device)

    # Test Training Forward Pass (ArcFace Head)
    # Requires labels
    logits = model(images.to(device), labels.to(device))
    assert logits.shape == (
        Config.BATCH_SIZE,
        len(classes),
    ), f"Logits shape mismatch. Expected {(Config.BATCH_SIZE, len(classes))}, got {logits.shape}"

    # Test Inference Forward Pass (Embeddings)
    # Labels = None
    embeddings = model(images.to(device), labels=None)
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_SIZE,
    ), f"Embedding shape mismatch. Expected {(Config.BATCH_SIZE, Config.EMBEDDING_SIZE)}, got {embeddings.shape}"

    print("    Model architecture verified.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Run training using the library function
    # This will save the model to Config.MODEL_PATH
    run_training(
        debug=Config.DEBUG, load_cached_data=True, epochs=Config.EPOCHS, patience=1
    )

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH} after training."

    print(f"    Training complete. Checkpoint saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # Run inference using the library function
    predict_and_submit(
        model_path=Config.MODEL_PATH,
        output_file=Config.SUBMISSION_FILE,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Verify content format
    df = pd.read_csv(Config.SUBMISSION_FILE)
    assert "image" in df.columns, "Submission missing 'image' column"
    assert "hotel_id" in df.columns, "Submission missing 'hotel_id' column"
    assert len(df) > 0, "Submission file is empty"

    # Check prediction format (space-delimited string)
    sample_pred = df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    assert (
        len(sample_pred.split()) == 5
    ), f"Expected 5 predictions per image, got {len(sample_pred.split())}"

    print(f"    Submission generated successfully at {Config.SUBMISSION_FILE}")
    print(f"    Sample prediction: {df.iloc[0]['image']} -> {sample_pred}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
