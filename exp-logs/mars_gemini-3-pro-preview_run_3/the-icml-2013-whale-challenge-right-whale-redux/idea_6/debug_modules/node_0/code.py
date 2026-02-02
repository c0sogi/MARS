import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
# We assume the script is executed from the project root
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNetV2
from library.trainer import Trainer


def run_demo():
    print("============================================================")
    print("       Whale Call Detection Pipeline Demo Execution")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config for a fast debug run
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PRETRAINED = False  # Disable downloading weights for speed

    # Set a specific working directory for this demo
    Config.IDEA_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join(Config.PROJECT_ROOT, "working", Config.IDEA_NAME)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Device: {Config.DEVICE}")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[2] Loading and processing data...")

    # We force `load_cached_data=False` to demonstrate the audio processing pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"    Train Batch Images Shape: {images.shape}")
    print(f"    Train Batch Labels Shape: {labels.shape}")

    # Validate Shapes
    # Expected Time Dimension: (Sample Rate * Duration) // Hop Length + 1
    # (2000 * 2.0) // 20 + 1 = 201
    expected_time_dim = (
        int(Config.SAMPLE_RATE * Config.DURATION) // Config.HOP_LENGTH + 1
    )

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.N_MELS,
        expected_time_dim,
    ), f"Image shape mismatch. Expected ({Config.BATCH_SIZE}, 3, {Config.N_MELS}, {expected_time_dim}), got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected ({Config.BATCH_SIZE},), got {labels.shape}"

    print("    Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass Check
    # -------------------------------------------------------------------------
    print("\n[3] Initializing model and checking forward pass...")

    model = WhaleEfficientNetV2(pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)
    model.eval()

    # Run dummy input through the model
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        # Use mixed precision context if on CUDA, matching Trainer logic
        if Config.DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                output = model(dummy_input)
        else:
            output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Validate Output Shape (Batch, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {output.shape}"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Starting training loop...")

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Execute Training
    best_auc = trainer.fit()

    # Validate Training Result
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    assert 0 <= best_auc <= 1, f"Invalid AUC score: {best_auc}"

    print(f"    Training finished successfully.")
    print(f"    Best Validation AUC: {best_auc:.4f}")
    print(f"    Checkpoint saved to: {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Running inference and generating submission...")

    # Load the best state dict
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    predictions = []
    clip_names = []

    with torch.no_grad():
        for batch_images, batch_names in test_loader:
            batch_images = batch_images.to(Config.DEVICE)

            if Config.DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(batch_images)
                    probs = torch.sigmoid(logits)
            else:
                logits = model(batch_images)
                probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())
            clip_names.extend(batch_names)

    # Create DataFrame
    submission_df = pd.DataFrame({"clip": clip_names, "probability": predictions})

    print(f"    Generated predictions for {len(submission_df)} samples.")
    print(f"    First 3 rows:\n{submission_df.head(3)}")

    # Validate Submission Format
    assert len(submission_df) == len(
        test_loader.dataset
    ), "Submission row count does not match test dataset size."
    assert list(submission_df.columns) == [
        "clip",
        "probability",
    ], "Submission columns do not match required format."

    # Save to file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")
    print("\n============================================================")
    print("       Demo Execution Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
