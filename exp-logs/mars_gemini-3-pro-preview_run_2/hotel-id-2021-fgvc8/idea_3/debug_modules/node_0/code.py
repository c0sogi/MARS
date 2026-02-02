import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, apk, mapk
from library.dataset import get_dataloaders, get_label_mapping
from library.model import HotelModel
from library.engine import run_training
from library.inference import predict


def main():
    # ==========================================
    # 1. Configuration Setup for Demonstration
    # ==========================================
    print(">>> Setting up configuration for demo run...")

    # Enable Debug mode to use small data subsets
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small sample for speed

    # Training Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 12
    Config.NUM_WORKERS = 2

    # Use a lightweight backbone for the demo to ensure it runs quickly on CPU/GPU
    Config.BACKBONE_NAME = "resnet18"
    Config.EMBEDDING_DIM = 128  # Reduced dimension for demo

    # Set up a specific working directory for this demo to avoid overwriting production files
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the demo working directory
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Update caching paths
    Config.GALLERY_EMB_PATH = os.path.join(Config.WORKING_DIR, "gallery.npy")
    Config.GALLERY_LABELS_PATH = os.path.join(Config.WORKING_DIR, "gallery_labels.npy")
    Config.QUERY_EMB_PATH = os.path.join(Config.WORKING_DIR, "query.npy")

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration configured successfully.")

    # ==========================================
    # 2. Metric Verification
    # ==========================================
    print("\n>>> Verifying Metrics (AP@K)...")
    # Test Case: Ground Truth=1, Predictions=[1, 2, 3, 4, 5] -> Score should be 1.0
    score_perfect = apk([1], [1, 2, 3, 4, 5], k=5)
    assert score_perfect == 1.0, f"Expected 1.0, got {score_perfect}"

    # Test Case: Ground Truth=1, Predictions=[2, 3, 4, 5, 1] -> Score should be 1/5 = 0.2
    score_last = apk([1], [2, 3, 4, 5, 1], k=5)
    assert score_last == 0.2, f"Expected 0.2, got {score_last}"
    print("Metrics verification passed.")

    # ==========================================
    # 3. Data Loading Verification
    # ==========================================
    print("\n>>> Verifying Data Loading...")
    # Generate dataloaders with debug=True
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(debug=True)

    # Check if loaders are populated
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Val loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions for shape
    # Shape: (Batch_Size, Channels, Height, Width)
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Channel count mismatch (should be 3 for RGB)"
    assert images.shape[2] == Config.IMAGE_SIZE, "Image height mismatch"
    assert images.shape[3] == Config.IMAGE_SIZE, "Image width mismatch"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print(f"Data Loading verification passed. Num Classes: {num_classes}")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n>>> Verifying Model Architecture...")
    # Instantiate the model
    # Note: Config.NUM_CLASSES is used by default in Model __init__
    model = HotelModel(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    # Test Forward Pass (Training Mode - returns logits)
    logits = model(images, labels)
    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Logits shape mismatch"

    # Test Forward Pass (Inference Mode - returns embeddings)
    embeddings = model(images, labels=None)
    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_DIM,
    ), "Embeddings shape mismatch"

    print("Model architecture verification passed.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n>>> Running Training Loop (1 Epoch)...")
    # Run training using the engine
    run_training(
        model, train_loader, val_loader, num_epochs=Config.EPOCHS, device=Config.DEVICE
    )

    # Verify that model artifacts were saved
    assert os.path.exists(Config.MODEL_PATH), "Final model file was not saved"
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved"
    print("Training loop executed successfully.")

    # ==========================================
    # 6. Inference and Submission Verification
    # ==========================================
    print("\n>>> Running Inference and Submission Generation...")

    # Run prediction pipeline
    # We set load_cached_data=False to force generation of embeddings for this demo run
    predict(
        model, test_loader, device=Config.DEVICE, load_cached_data=False, debug=True
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV was not created"

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{sub_df.head()}")

    assert len(sub_df) > 0, "Submission file is empty"
    assert "image" in sub_df.columns, "Submission missing 'image' column"
    assert "hotel_id" in sub_df.columns, "Submission missing 'hotel_id' column"

    # Check prediction format (space-delimited list of 5 IDs)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction must be a string"
    pred_list = sample_pred.split()
    assert len(pred_list) == 5, f"Expected 5 predictions, got {len(pred_list)}"

    print("Inference and Submission verification passed.")
    print("\n>>> All demonstrations completed successfully!")


if __name__ == "__main__":
    main()
