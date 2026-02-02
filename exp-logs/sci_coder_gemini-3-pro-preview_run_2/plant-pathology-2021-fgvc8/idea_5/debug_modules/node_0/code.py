import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_f1_score
from library.data import get_loaders
from library.models import AppleDiseaseModel
from library.engine import train_one_epoch, validate
from library.inference import load_model_for_inference, predict_ensemble


def run_demo():
    print("=== Starting Apple Disease Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True  # Use small subset of data (100 samples)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.MODEL_1_NAME = "resnet18"  # Use lightweight model for speed verification

    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Model Architecture: {Config.MODEL_1_NAME}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Initializing Data Loaders...")

    # Get loaders (this will use the pre-generated metadata in ./metadata)
    # load_cached_data=False forces a fresh read from CSVs for this demo
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify Train Loader
    try:
        images, targets, image_ids = next(iter(train_loader))
        print(f"    Train Batch - Images: {images.shape}, Targets: {targets.shape}")

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Incorrect image shape"
        assert targets.shape == (
            Config.BATCH_SIZE,
            Config.NUM_CLASSES,
        ), "Incorrect target shape"
        assert len(image_ids) == Config.BATCH_SIZE, "Incorrect number of image IDs"
        assert targets.dtype == torch.float32, "Targets should be float32"

        print("    Data Loader verification passed.")
    except StopIteration:
        print("    Error: Train loader is empty!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")

    # Initialize model (pretrained=False to avoid download overhead/errors in this env)
    model = AppleDiseaseModel(model_name=Config.MODEL_1_NAME, pretrained=False)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        output = model(dummy_input)

    print(f"    Output Logits Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("    Model initialization and forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Setup Optimizer and Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Train for one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, device, scaler, epoch=0
    )
    print(f"    Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_loss, val_score = validate(model, val_loader, device)
    print(f"    Val Loss: {val_loss:.4f}, Val F1 Score: {val_score:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_score <= 1.0, "F1 Score out of range"

    print("    Training and validation loop verification passed.")

    # -------------------------------------------------------------------------
    # 5. Checkpointing & Inference Preparation
    # -------------------------------------------------------------------------
    print("\n[5] Testing Checkpoint Saving and Loading...")

    # Save Checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    # Manually saving to mimic save_checkpoint logic but with custom name for demo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "score": val_score,
        },
        checkpoint_path,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"

    # Load Model for Inference
    loaded_model = load_model_for_inference(
        Config.MODEL_1_NAME, checkpoint_path, device
    )
    assert loaded_model is not None, "Failed to load model from checkpoint"

    # Verify weights match (simple check on first parameter)
    p1 = next(model.parameters())
    p2 = next(loaded_model.parameters())
    assert torch.allclose(p1, p2), "Loaded weights do not match saved weights"

    print("    Checkpoint save/load verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Run prediction
    # Using TTA=False for speed in demo
    df_submission = predict_ensemble(
        models=[loaded_model],
        loader=test_loader,
        device=device,
        threshold=0.5,
        use_tta=False,
    )

    # Verify Output
    print(f"    Inference Result Shape: {df_submission.shape}")
    print("    Head of Predictions:")
    print(df_submission.head())

    # Assertions
    assert "image" in df_submission.columns, "Missing 'image' column"
    assert "labels" in df_submission.columns, "Missing 'labels' column"
    assert len(df_submission) == len(test_loader.dataset), "Prediction count mismatch"

    # Check label format (should be string, space delimited or 'healthy')
    if len(df_submission) > 0:
        sample_label = df_submission.iloc[0]["labels"]
        assert isinstance(sample_label, str), "Labels must be strings"

    # Simulate saving submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(sub_path, index=False)
    assert os.path.exists(sub_path), "Submission file not saved"

    print("    Inference and submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
