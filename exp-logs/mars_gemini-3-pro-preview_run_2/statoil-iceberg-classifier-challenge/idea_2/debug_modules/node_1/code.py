import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import modules from the provided library
from library.utils import load_dataset, seed_everything
from library.dataset import get_dataloaders
from library.model import SAHCN
from library.train import run_training_pipeline
from library.predict import generate_predictions


def run_demo():
    print("==================================================")
    print("   Iceberg Classifier Library Demonstration       ")
    print("==================================================")

    # Define temporary paths for this demo to avoid overwriting production files
    demo_work_dir = "./working/demo_artifacts"
    os.makedirs(demo_work_dir, exist_ok=True)

    demo_model_path = os.path.join(demo_work_dir, "demo_model.pth")
    demo_sub_path = os.path.join(demo_work_dir, "demo_submission.csv")

    # ---------------------------------------------------------
    # 1. Reproducibility
    # ---------------------------------------------------------
    print("\n[Step 1] Setting random seeds...")
    seed_everything(42)
    print("Seeds set successfully.")

    # ---------------------------------------------------------
    # 2. Data Loading & Inspection
    # ---------------------------------------------------------
    print("\n[Step 2] Loading and inspecting dataset...")
    # Load data (uses cache if available for speed)
    data = load_dataset(load_cached_data=True)

    # Verify essential keys exist
    required_keys = ["X_train", "angle_train", "y_train", "X_test"]
    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Dataset dictionary missing key: {key}")

    # Verify Image Shape: (N, 3, 75, 75)
    # The library converts 2-band data into 3 channels
    x_train_shape = data["X_train"].shape
    if len(x_train_shape) != 4 or x_train_shape[1:] != (3, 75, 75):
        raise AssertionError(
            f"Unexpected X_train shape: {x_train_shape}. Expected (N, 3, 75, 75)."
        )

    print(f"Data loaded. Training samples: {x_train_shape[0]}")

    # ---------------------------------------------------------
    # 3. DataLoader Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying DataLoaders (Debug Mode)...")
    debug_size = 32
    batch_size = 8

    # Get loaders with debug flag to limit dataset size
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, debug=True, debug_size=debug_size
    )

    # Fetch one batch to check tensor shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    # Verify tensor dimensions
    if images.shape != (batch_size, 3, 75, 75):
        raise AssertionError(f"Batch image shape mismatch: {images.shape}")
    if angles.shape != (batch_size,):
        raise AssertionError(f"Batch angle shape mismatch: {angles.shape}")
    if labels.shape != (batch_size,):
        raise AssertionError(f"Batch label shape mismatch: {labels.shape}")

    print("DataLoader yielded correct tensor shapes.")

    # ---------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 4] Testing Model Architecture (SAHCN)...")
    model = SAHCN(dropout_rate=0.5)

    # Run a forward pass on CPU to verify architecture
    model.eval()
    with torch.no_grad():
        outputs = model(images, angles)

    # Output should be (Batch_Size, 1) raw logits
    if outputs.shape != (batch_size, 1):
        raise AssertionError(
            f"Model output shape mismatch: {outputs.shape}. Expected ({batch_size}, 1)."
        )

    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Pipeline
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Pipeline...")
    # Run a minimal training loop (1 epoch)
    run_training_pipeline(
        load_cached_data=True,
        batch_size=8,
        learning_rate=0.001,
        num_epochs=1,  # Limit epochs for speed
        patience=1,
        dropout_rate=0.5,
        debug=True,  # Use small subset
        debug_size=debug_size,
        model_save_path=demo_model_path,
        submission_path=demo_sub_path,
    )

    # Verify model artifact creation
    if not os.path.exists(demo_model_path):
        raise AssertionError(f"Model file was not created at {demo_model_path}")

    print(f"Training complete. Model saved to {demo_model_path}")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[Step 6] Running Prediction Pipeline...")

    # Delete the submission created by the training pipeline to verify the prediction pipeline works independently
    if os.path.exists(demo_sub_path):
        os.remove(demo_sub_path)

    # Generate predictions using the saved model
    generate_predictions(
        model_path=demo_model_path,
        submission_path=demo_sub_path,
        batch_size=8,
        debug=True,
        debug_size=debug_size,
        load_cached_data=True,
    )

    # Verify submission file
    if not os.path.exists(demo_sub_path):
        raise AssertionError(f"Submission file was not created at {demo_sub_path}")

    # Verify submission content
    df_sub = pd.read_csv(demo_sub_path)
    if list(df_sub.columns) != ["id", "is_iceberg"]:
        raise AssertionError(f"Submission columns incorrect: {df_sub.columns}")

    # In debug mode, test set is truncated to debug_size
    if len(df_sub) != debug_size:
        raise AssertionError(
            f"Submission row count mismatch. Expected {debug_size}, got {len(df_sub)}"
        )

    print(f"Prediction complete. Submission saved to {demo_sub_path}")
    print("\nAll demonstration steps passed successfully!")


if __name__ == "__main__":
    run_demo()
