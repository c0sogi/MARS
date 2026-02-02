import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, count_parameters, load_checkpoint
from library.data_loader import (
    load_and_process_data,
    create_fold_loaders,
    create_test_loader,
)
from library.model import SpatiallyRegularizedSECNN
from library.trainer import train_fold, validate


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo.
    print("\n[Step 1] Configuring environment...")

    # Set a specific working directory for this demo
    Config.WORK_DIR = "./working/demo_usage"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.create_directories()

    # Enable Debug mode to use a tiny subset of data (50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50

    # Reduce training intensity
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 3  # Reduced folds (we will only run fold 0)

    # Set seed for reproducibility
    set_seed(42)
    print("Configuration updated for fast execution.")

    # 2. Data Loading
    print("\n[Step 2] Loading and processing data...")
    # This will load raw JSONs, process them, and cache them (or load from cache if exists)
    # We force debug=True to get the small subset
    X_train, angles_train, y_train, X_test, angles_test, ids_test = (
        load_and_process_data(
            debug=Config.DEBUG,
            load_cached_data=False,  # Force processing to demonstrate the pipeline
        )
    )

    # Validation of Data Shapes
    print(f"Data Loaded: {len(X_train)} training samples, {len(X_test)} test samples.")

    # Assertions to ensure data integrity
    assert len(X_train) == Config.DEBUG_SAMPLES, "X_train size mismatch"
    assert X_train.shape[1:] == (3, 75, 75), f"Unexpected image shape: {X_train.shape}"
    assert len(angles_train) == len(X_train), "Angle count mismatch"
    assert len(y_train) == len(X_train), "Label count mismatch"

    # 3. DataLoader Creation
    print("\n[Step 3] Creating DataLoaders for Fold 0...")
    train_loader, val_loader = create_fold_loaders(
        X_train, angles_train, y_train, fold_idx=0
    )

    # Verify a single batch
    images, angles, labels = next(iter(train_loader))
    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)

    # 4. Model Instantiation
    print("\n[Step 4] Instantiating Model...")
    device = Config.DEVICE
    print(f"Device: {device}")

    model = SpatiallyRegularizedSECNN().to(device)

    # Check parameter count
    params = count_parameters(model)
    print(f"Model created with {params:,} trainable parameters.")

    # Verify Forward Pass logic
    with torch.no_grad():
        # Use the batch we fetched earlier
        dummy_out = model(images.to(device), angles.to(device))
        assert dummy_out.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
        print("Forward pass verification successful.")

    # 5. Training Loop
    print("\n[Step 5] Training Fold 0...")
    # train_fold handles the loop, validation, and checkpoint saving
    best_loss = train_fold(0, model, train_loader, val_loader, device)
    print(f"Training completed. Best Validation Loss: {best_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # 6. Inference / Prediction
    print("\n[Step 6] Running Inference on Test Set...")

    # Load the best model
    best_model = SpatiallyRegularizedSECNN().to(device)
    checkpoint = load_checkpoint(checkpoint_path, best_model)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}.")

    # Create Test Loader
    test_loader = create_test_loader(X_test, angles_test, ids_test)

    # We can reuse the validate function logic, but since validate expects labels,
    # we'll write a quick inference snippet here using the model directly.
    best_model.eval()
    predictions = []

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = best_model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.extend(probs)

    predictions = np.array(predictions)

    # Validate Predictions
    assert len(predictions) == len(
        ids_test
    ), "Prediction count does not match test set size."
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]."

    print(f"Generated {len(predictions)} predictions.")
    print(f"Sample predictions: {predictions[:5]}")

    # 7. Generate Submission File
    print("\n[Step 7] Saving Demo Submission...")
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": predictions})

    out_file = os.path.join(Config.WORK_DIR, "demo_submission.csv")
    submission_df.to_csv(out_file, index=False)
    print(f"Submission saved to {out_file}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
