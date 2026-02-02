import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, log_message
from library.dataset import (
    get_train_val_loaders,
    get_test_loader,
    get_pseudo_label_loader,
    load_and_process_data,
)
from library.model import IcebergResNet18
from library.engine import run_swa_training
from library.inference import predict_with_tta, create_submission


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # --------------------------------------------------------------------------
    print(">>> Step 1: Configuring environment for demo execution...")

    # Override Config paths to use a separate demo directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_TRAIN_IMAGES = os.path.join(demo_dir, "train_images.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "train_labels.npy")
    Config.CACHE_TRAIN_ANGLES = os.path.join(demo_dir, "train_angles.npy")
    Config.CACHE_TEST_IMAGES = os.path.join(demo_dir, "test_images.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "test_ids.npy")
    Config.CACHE_TEST_ANGLES = os.path.join(demo_dir, "test_angles.npy")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = (
        demo_dir  # Save submission directly in demo dir for simplicity
    )
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n>>> Step 2: Verifying Data Loading...")
    # Load loaders (this will process JSONs and cache them in the demo dir)
    train_loader, val_loader = get_train_val_loaders(load_cached_data=True)

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Batch Shape: {images.shape}")
    print(f"Angle Batch Shape: {angles.shape}")
    print(f"Label Batch Shape: {labels.shape}")

    # Assertions
    assert len(images.shape) == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == 224 and images.shape[3] == 224
    ), "Images should be resized to 224x224"
    assert len(angles.shape) == 1, "Angles should be 1D tensor"
    assert len(labels.shape) == 1, "Labels should be 1D tensor"
    print("Data Loading verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # --------------------------------------------------------------------------
    print("\n>>> Step 3: Verifying Model Architecture...")
    model = IcebergResNet18()
    model.to(device)

    # Move dummy batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(images, angles)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (images.size(0), 1), "Output shape should be (Batch_Size, 1)"
    print("Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration (Phase 1 + SWA)
    # --------------------------------------------------------------------------
    print("\n>>> Step 4: Demonstrating Training Loop (Fast Mode)...")
    # We use a very short schedule: 1 epoch standard, 1 epoch SWA
    swa_start = 1
    total_epochs = 2

    run_swa_training(
        model,
        train_loader,
        val_loader,
        device=device,
        swa_start_epoch=swa_start,
        total_epochs=total_epochs,
        patience=2,
        save_path_best=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
        save_path_swa=os.path.join(Config.CHECKPOINT_DIR, "swa_model.pth"),
    )

    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    ), "Best model checkpoint not found"
    # SWA model might not be saved if we stop early or if logic dictates, but with 2 epochs it should run.
    # Note: run_swa_training logic: if epoch <= swa_start (1 <= 1), run standard.
    # epoch 2: else run SWA.
    # So SWA runs at epoch 2.
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "swa_model.pth")
    ), "SWA model checkpoint not found"
    print("Training loop executed successfully.")

    # --------------------------------------------------------------------------
    # 5. Inference and TTA
    # --------------------------------------------------------------------------
    print("\n>>> Step 5: Verifying Inference with TTA...")
    test_loader = get_test_loader(load_cached_data=True)

    # Use the trained model for prediction
    predictions = predict_with_tta(model, test_loader, device=device)

    # Check results
    test_ids = [
        data[2] for data in test_loader.dataset
    ]  # Access dataset directly to get IDs
    print(f"Number of predictions: {len(predictions)}")
    print(f"Sample Prediction: {list(predictions.items())[0]}")

    # Assertions
    assert len(predictions) > 0, "Predictions dictionary is empty"
    assert isinstance(
        list(predictions.values())[0], float
    ), "Prediction values should be floats"
    print("Inference verified successfully.")

    # --------------------------------------------------------------------------
    # 6. Semi-Supervised Logic Verification
    # --------------------------------------------------------------------------
    print("\n>>> Step 6: Verifying Pseudo-Labeling Logic...")

    # Since the model is barely trained, it might not produce high confidence predictions.
    # We manually create a dummy prediction dictionary to test the loader logic.
    # We'll take the first few test IDs and assign them extreme probabilities.

    # Get actual test IDs
    _, _, test_ids_array = load_and_process_data(mode="test", load_cached_data=True)

    dummy_preds = {}
    # Make 5 high confidence icebergs (1.0) and 5 high confidence ships (0.0)
    for i in range(min(10, len(test_ids_array))):
        tid = test_ids_array[i]
        if i % 2 == 0:
            dummy_preds[tid] = 0.99  # > 0.95 (Iceberg)
        else:
            dummy_preds[tid] = 0.01  # < 0.05 (Ship)

    print(f"Created {len(dummy_preds)} dummy high-confidence predictions.")

    # Create combined loader
    combined_loader = get_pseudo_label_loader(dummy_preds, load_cached_data=True)

    # Check size
    original_size = len(train_loader.dataset)
    combined_size = len(combined_loader.dataset)
    print(f"Original Train Size: {original_size}")
    print(f"Combined Train Size: {combined_size}")

    # Assertions
    # We added 10 samples, so combined size should be original + 10
    assert (
        combined_size == original_size + 10
    ), f"Expected {original_size + 10} samples, got {combined_size}"

    # Verify we can iterate
    c_images, c_angles, c_labels = next(iter(combined_loader))
    assert c_images.shape[0] == Config.BATCH_SIZE, "Combined loader batch size mismatch"
    print("Pseudo-labeling logic verified successfully.")

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    print("\n>>> Step 7: Generating Submission...")
    create_submission(predictions, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    print("Submission generation verified successfully.")

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    run_demo()
