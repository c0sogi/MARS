import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.models import PathologyClassifier
from library.training import Trainer, set_seed
from library.inference import generate_submission, InferenceEngine


def run_demo():
    print("Starting End-to-End Pipeline Demonstration...")

    # --- 1. Configuration Overrides for Speed ---
    # We modify the Config class attributes directly to create a fast debug run
    print("\n[1/5] Configuring environment for demo execution...")

    Config.PROJECT_NAME = "demo_execution"
    Config.DEBUG = True
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    Config.MODEL_BACKBONES = ["convnext_tiny"]  # Use single small backbone

    # Re-define working directories based on new project name
    Config.WORK_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")

    # Clean up previous demo runs if they exist to ensure a fresh start
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.print_config()
    set_seed(Config.SEED)

    # --- 2. Data Loading Verification ---
    print("\n[2/5] Verifying Data Loading...")

    # Use a small debug_size to load only a subset of data
    DEBUG_SIZE = 100
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # Fetch a single batch to verify shapes
    images, labels, ids = next(iter(train_loader))

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Label Shape: {labels.shape}")

    # Assertions
    expected_shape = (Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch."
    assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch."

    print("  Data Loading logic verified successfully.")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n[3/5] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = PathologyClassifier(
        model_name=Config.MODEL_BACKBONES[0],
        num_classes=1,
        pretrained=False,  # False for speed/offline safety in demo
    ).to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, Config.CROP_SIZE, Config.CROP_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("  Model architecture verified successfully.")

    # --- 4. Training Loop Execution ---
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        fold_idx=0,
    )

    # Run training
    trainer.fit(epochs=Config.EPOCHS)

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(
        Config.CHECKPOINT_DIR, f"best_model_{Config.MODEL_BACKBONES[0]}_fold_0.pth"
    )

    # If validation AUC didn't improve (rare but possible in 1 epoch with random init),
    # the 'last_model' should definitely exist.
    last_checkpoint = os.path.join(
        Config.CHECKPOINT_DIR, f"last_model_{Config.MODEL_BACKBONES[0]}_fold_0.pth"
    )

    assert os.path.exists(expected_checkpoint) or os.path.exists(
        last_checkpoint
    ), "Training failed to generate a checkpoint file."

    # Ensure we have a 'best_model' for the inference step.
    # If best_model wasn't saved (e.g. val auc 0), copy last to best for demo purposes.
    if not os.path.exists(expected_checkpoint):
        shutil.copy(last_checkpoint, expected_checkpoint)

    print("  Training loop completed and checkpoint verified.")

    # --- 5. Inference & Submission ---
    print("\n[5/5] Running Inference and Generating Submission...")

    # We use the generate_submission function which handles the full inference pipeline
    # We pass the same debug_size so it uses the test subset
    generate_submission(load_cached_data=False, debug_size=DEBUG_SIZE)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission loaded. Rows: {len(df_sub)}")
    print(df_sub.head(3))

    # Assertions on submission content
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns."
    assert (
        len(df_sub) == DEBUG_SIZE
    ), f"Submission row count mismatch. Expected {DEBUG_SIZE}, got {len(df_sub)}"
    assert df_sub["label"].dtype == float, "Label column should be float probabilities."

    print("  Inference pipeline verified successfully.")
    print("\nAll demonstration steps passed!")


if __name__ == "__main__":
    run_demo()
