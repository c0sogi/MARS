import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_dataloaders
from library.models import get_model
from library.trainer import run_training


def main():
    print("Initializing Demo Execution...")

    # 1. Override Configuration for Fast Demonstration
    # We modify the Config class attributes directly to create a lightweight run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples per split
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo stability
    Config.WORKING_DIR = "./working/demo_execution"  # Dedicated working directory

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Set Random Seeds for Reproducibility
    seed_everything(Config.SEED)

    # 3. Data Loading
    print("\n--- Data Loading ---")
    # We force reload (load_cached_data=False) to demonstrate processing logic
    # and ensure we don't pick up stale cache files from other runs.
    loaders = get_dataloaders(debug=True, load_cached_data=False)

    # Validate DataLoaders
    assert "train" in loaders and "val" in loaders and "test" in loaders

    # Validate Train Batch Structure
    train_loader = loaders["train"]
    images, targets = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Targets Shape: {targets.shape}")

    # Expected: (Batch, Channels, Freq, Time)
    # Channels=1, Freq=128 (N_MELS), Time approx 63 (4000/64)
    assert images.dim() == 4
    assert images.shape[1] == 1
    assert targets.dim() == 1

    # 4. Model Initialization
    print("\n--- Model Initialization ---")
    device = Config.DEVICE
    model_name = "resnet34"  # Using ResNet34 as the demo model

    model = get_model(model_name, pretrained=True)
    model = model.to(device)

    # Validate Model Forward Pass
    # Use the real batch from the dataloader to ensure dimension compatibility
    with torch.no_grad():
        output = model(images.to(device))

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1)  # Binary classification logits

    # 5. Training Loop
    print("\n--- Training Loop ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run training for Fold 0
    # This handles training, validation, logging, and checkpoint saving
    best_score = run_training(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        fold_idx=0,
        model_name_suffix="demo_model",
    )

    print(f"Training finished. Best Validation Score: {best_score}")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_model_fold_0.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Expected checkpoint file not found at {checkpoint_path}"
        )

    # 6. Inference and Submission
    print("\n--- Inference & Submission ---")

    # Load the best model weights
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    test_loader = loaders["test"]
    all_probs = []
    all_clips = []

    with torch.no_grad():
        for images, clips in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_clips.extend(clips)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"clip": all_clips, "probability": all_probs})

    # Validate Submission Format
    print(f"Generated predictions for {len(submission_df)} clips.")
    assert len(submission_df) == len(
        pd.read_csv(Config.TEST_CSV).head(Config.DEBUG_SUBSET_SIZE)
    )
    assert submission_df["probability"].min() >= 0.0
    assert submission_df["probability"].max() <= 1.0

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
