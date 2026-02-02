import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for library imports
sys.path.append(".")

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.models import get_model
from library.loss import AsymmetricLoss
from library.train import train_specific_model
from library.inference import generate_submission
from library.utils import calculate_f1_score


def run_demonstration():
    print("--- Starting Pipeline Demonstration ---")

    # 1. Configuration Overrides for Speed
    # We monkey-patch the Config class to run a fast, minimal version of the pipeline.
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 8  # Smaller batch size for the demo
    Config.NUM_WORKERS = 2  # Reduce overhead

    # For the demo, we use the same architecture for Model B to avoid training a second model from scratch
    # or needing a compatible checkpoint for a different architecture.
    Config.MODEL_B_NAME = Config.MODEL_A_NAME

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    # Force reload cache to ensure we test the processing logic on the subset
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify Train Loader
    images, targets, ids = next(iter(train_loader))
    print(f"Train Batch - Image Shape: {images.shape}, Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Incorrect image tensor shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect target tensor shape"
    assert isinstance(ids[0], str), "IDs should be strings"

    print("Data loading logic verified.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")
    model_name = Config.MODEL_A_NAME  # e.g., resnet101d
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)
    model.eval()

    # Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model architecture verified.")

    # Clean up
    del model, dummy_input, output
    torch.cuda.empty_cache()

    # 4. Loss Function Verification
    print("\n[Step 4] Verifying Loss Function...")
    criterion = AsymmetricLoss()

    # Create dummy logits and targets
    dummy_logits = torch.randn(4, Config.NUM_CLASSES, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (4, Config.NUM_CLASSES)).float()

    loss = criterion(dummy_logits, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # backward pass check
    loss.backward()
    assert dummy_logits.grad is not None, "Gradients not computed"
    print("Loss function verified.")

    # 5. Training Loop Demonstration
    print("\n[Step 5] Demonstrating Training Loop (Model A)...")
    # We train MODEL_A for 1 epoch on the debug subset
    # This function handles optimizer, scheduler, scaler, and saving checkpoints
    best_score_a = train_specific_model(
        Config.MODEL_A_NAME, epochs=Config.EPOCHS, debug=Config.DEBUG
    )

    checkpoint_path_a = os.path.join(
        Config.WORKING_DIR, f"{Config.MODEL_A_NAME}_best.pth"
    )
    assert os.path.exists(
        checkpoint_path_a
    ), f"Checkpoint not found at {checkpoint_path_a}"
    print(f"Training Model A completed. Best Score: {best_score_a}")

    # 6. Mocking Model B Training (to save time)
    # For the ensemble demonstration in the next step, we need a checkpoint for Model B.
    # We will just copy Model A's checkpoint to Model B's path to simulate that Model B was trained.
    print("\n[Step 6] Simulating Model B Training (Copying Model A checkpoint)...")
    checkpoint_path_b = os.path.join(
        Config.WORKING_DIR, f"{Config.MODEL_B_NAME}_best.pth"
    )
    if checkpoint_path_a != checkpoint_path_b:
        shutil.copy(checkpoint_path_a, checkpoint_path_b)

    assert os.path.exists(checkpoint_path_b), "Model B checkpoint creation failed"
    print("Model B checkpoint ready.")

    # 7. Inference and Submission Generation
    print("\n[Step 7] Generating Submission (Ensemble Inference)...")
    # This function loads val/test data, loads models A and B, predicts, optimizes threshold, and writes CSV
    val_f1 = generate_submission(debug=Config.DEBUG, load_cached_data=False)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    assert "id" in sub_df.columns, "Missing 'id' column"
    assert "attribute_ids" in sub_df.columns, "Missing 'attribute_ids' column"
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(sub_df)}"

    # Verify content format (space separated integers or empty)
    # Just check the first row
    first_attr = sub_df.iloc[0]["attribute_ids"]
    if isinstance(first_attr, str) and len(first_attr) > 0:
        parts = first_attr.split()
        assert all(p.isdigit() for p in parts), "attribute_ids should contain integers"

    print("Submission generation verified.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
