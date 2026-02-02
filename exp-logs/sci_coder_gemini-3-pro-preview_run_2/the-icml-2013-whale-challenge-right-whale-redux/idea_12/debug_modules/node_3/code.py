import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data_loader
from library.models import WhaleClassifier
from library.engine import train_fn, inference_fn


def create_subset_metadata(source_csv, dest_csv, n_samples):
    """
    Reads the source metadata CSV, samples n_samples, and saves to dest_csv.
    """
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source metadata not found: {source_csv}")

    df = pd.read_csv(source_csv)
    # Ensure we don't sample more than available
    n = min(n_samples, len(df))
    df_subset = df.sample(n=n, random_state=Config.SEED).reset_index(drop=True)
    df_subset.to_csv(dest_csv, index=False)
    print(f"Created subset metadata at {dest_csv} with {n} samples.")


def run_demo():
    # 1. Setup and Configuration
    print("--- Setting up Configuration for Demo ---")
    seed_everything(Config.SEED)

    # Define a temporary working directory for this demo
    demo_working_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = demo_working_dir
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce workers for small batch

    # Create subset metadata files to avoid processing the entire dataset
    mini_train_path = os.path.join(demo_working_dir, "train_mini.csv")
    mini_val_path = os.path.join(demo_working_dir, "val_mini.csv")
    mini_test_path = os.path.join(demo_working_dir, "test_mini.csv")

    create_subset_metadata(Config.TRAIN_CSV, mini_train_path, n_samples=50)
    create_subset_metadata(Config.VAL_CSV, mini_val_path, n_samples=20)
    create_subset_metadata(Config.TEST_CSV, mini_test_path, n_samples=20)

    # Point Config to these new files
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    print("Configuration updated for rapid execution.")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Pipeline ---")
    # This will trigger process_and_cache_data for the subsets
    train_loader = get_data_loader("train", batch_size=Config.BATCH_SIZE)
    val_loader = get_data_loader("val", batch_size=Config.BATCH_SIZE)

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Expected: (B, 1, 128, 4000) approx
    print(f"Batch Label Shape: {labels.shape}")  # Expected: (B,)

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensors (B, C, F, T)"
    assert images.shape[1] == 1, f"Expected 1 channel, got {images.shape[1]}"
    assert (
        images.shape[2] == Config.N_MELS
    ), f"Expected {Config.N_MELS} Mels, got {images.shape[2]}"
    assert labels.dim() == 1, "Labels should be 1D tensors"

    print("Data Pipeline verified successfully.")

    # 3. Model Initialization & Verification
    print("\n--- Verifying Model Architecture ---")
    model_name = "resnet34"
    model = WhaleClassifier(model_name=model_name, pretrained=True)
    model.to(Config.DEVICE)

    # Verify input adaptation (First conv layer should have 1 input channel)
    first_conv = model.backbone.conv1
    print(f"First Conv Layer: {first_conv}")
    assert (
        first_conv.in_channels == 1
    ), "Model first convolution not adapted for 1-channel input!"

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (images.shape[0], 1), "Model output shape mismatch!"

    print("Model architecture verified successfully.")

    # 4. Training Loop Execution
    print("\n--- Executing Training Loop (Demo) ---")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = nn.BCEWithLogitsLoss()

    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    best_auc = train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        criterion=criterion,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=save_path,
    )

    print(f"Training complete. Best Validation AUC: {best_auc:.4f}")
    assert 0.0 <= best_auc <= 1.0, "AUC score out of bounds!"
    assert os.path.exists(save_path), "Model checkpoint was not saved!"

    # 5. Inference & Submission
    print("\n--- Executing Inference ---")
    test_loader = get_data_loader("test", batch_size=Config.BATCH_SIZE)

    # Load best model for inference
    checkpoint = torch.load(save_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint)

    clips, probabilities = inference_fn(model, test_loader, Config.DEVICE)

    print(f"Inference complete. Generated {len(probabilities)} predictions.")

    # Verify consistency
    assert len(clips) == len(
        probabilities
    ), "Mismatch between clip names and predictions"
    assert len(clips) == 20, f"Expected 20 test samples, got {len(clips)}"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
