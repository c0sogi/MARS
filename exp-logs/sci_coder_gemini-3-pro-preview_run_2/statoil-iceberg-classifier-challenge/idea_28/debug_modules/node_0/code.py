import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import DIDPNet
from library.train import run_fold

if __name__ == "__main__":
    print("Initializing Demo Script for Ship vs Iceberg Classification...")

    # ==========================================
    # 1. CONFIGURATION FOR DEMO
    # ==========================================
    # Modify Config to run a fast demonstration
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_PATH = os.path.join(Config.WORK_DIR, "cache", "processed_data.npz")
    Config.MODEL_PATH_TEMPLATE = os.path.join(Config.WORK_DIR, "model_fold_{}.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")

    # Reduce compute load for demonstration
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.N_FOLDS = 2  # We will only run fold 0
    Config.setup()  # Ensure directories exist

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. DATA PIPELINE VERIFICATION
    # ==========================================
    print("\n[Step 1] Verifying Data Pipeline...")

    # Get dataloaders in debug mode (small subset of data)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        Config, fold_index=0, debug=True
    )

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors (N, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels (Band1, Band2, Mean)"
    assert images.shape[2] == 75 and images.shape[3] == 75, "Images must be 75x75"
    assert angles.dim() == 2 or angles.dim() == 1, "Angles must be 1D or 2D tensor"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and labels"
    print("Data Pipeline Verified.")

    # ==========================================
    # 3. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n[Step 2] Verifying Model Architecture...")

    model = DIDPNet(backbone_filters=32, dropout_rate=0.1).to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (images.shape[0], 1), "Output shape must be (Batch_Size, 1)"
    print("Model Architecture Verified.")

    # ==========================================
    # 4. TRAINING LOOP DEMONSTRATION
    # ==========================================
    print("\n[Step 3] Running Training Loop (Fold 0)...")

    # Run a single fold training session
    # This uses the library.train.run_fold function which handles the loop, validation, and saving
    val_loss = run_fold(fold_index=0, debug=True)

    print(f"Training completed. Best Validation Loss: {val_loss:.4f}")

    # Verify model file was created
    model_path = Config.MODEL_PATH_TEMPLATE.format(0)
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("Model checkpoint saved successfully.")

    # ==========================================
    # 5. INFERENCE & SUBMISSION
    # ==========================================
    print("\n[Step 4] Generating Submission...")

    # Load the best model
    best_model = DIDPNet(
        backbone_filters=Config.BACKBONE_FILTERS, dropout_rate=Config.DROPOUT_RATE
    )
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.to(device)
    best_model.eval()

    predictions = []

    # Inference Loop
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Get logits
            logits = best_model(images, angles)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())

    predictions = np.array(predictions)

    # Verify predictions
    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Probabilities must be between 0 and 1"

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())

    print("\nDemo execution completed successfully.")
