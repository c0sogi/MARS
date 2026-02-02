import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data_loader import get_train_val_loaders, get_test_loader, load_data
from library.model_factory import WhaleEfficientNet
from library.trainer import run_fold, validate_one_epoch


def main():
    print("=== Right Whale Detection Library Demo ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a small subset of data (500 samples)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Even smaller for this demo script

    # Reduce training duration
    Config.EPOCHS = 2
    Config.SWA_START_EPOCH = 1  # Start SWA immediately to test logic

    # Adjust batch size and workers for the demo environment
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Disable pretraining download to ensure offline execution speed
    Config.PRETRAINED = False

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=2, PRETRAINED=False")

    # ------------------------------------------------------------------------
    # 2. Data Loading Validation
    # ------------------------------------------------------------------------
    print("\n[2] Validating Data Loading Pipeline...")

    # Force reload to ensure we process the debug subset
    # This calls load_data internally
    train_loader, val_loader = get_train_val_loaders(fold_idx=0, load_cached_data=False)

    # Fetch a single batch to verify shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Verify Image Dimensions: (Batch, 1, Freq, Time)
    # Expected: (8, 1, 128, ~63) based on SR=2000, Dur=2.0, Hop=64
    assert images.ndim == 4, "Images must be 4D tensors (B, C, F, T)"
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}"
    assert images.shape[1] == 1, "Expected 1 channel (Grayscale Spectrogram)"
    assert images.shape[2] == Config.N_MELS, f"Expected {Config.N_MELS} Mel bands"

    # Verify Targets
    assert targets.ndim == 1, "Targets must be 1D tensors"
    assert targets.dtype == torch.float32, "Targets must be float32"

    print("Data Loader assertions passed.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Validation
    # ------------------------------------------------------------------------
    print("\n[3] Validating Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = WhaleEfficientNet(
        model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
    )
    model.to(device)

    # Perform a forward pass with the batch fetched earlier
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify Output Dimensions: (Batch, Num_Classes)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, {Config.NUM_CLASSES})"

    print("Model architecture assertions passed.")

    # ------------------------------------------------------------------------
    # 4. Training Loop & SWA Validation
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Fold 0)...")

    # Run the provided trainer function
    # This handles Optimizer, Scheduler, SWA, and Validation
    trained_model, best_auc = run_fold(
        fold_idx=0, train_loader=train_loader, val_loader=val_loader
    )

    print(f"Training completed. Final SWA Validation AUC: {best_auc:.4f}")

    # Verify Model Checkpoint was saved
    checkpoint_path = os.path.join(Config.WORKING_DIR, "swa_model_fold_0.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Model checkpoint not found at {checkpoint_path}"

    print("Training loop and checkpointing verified.")

    # ------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # ------------------------------------------------------------------------
    print("\n[5] Validating Inference on Test Set...")

    test_loader = get_test_loader(load_cached_data=True)
    trained_model.to(device)
    trained_model.eval()

    predictions = []
    clips = []

    # Run inference on a subset of test data (first batch only for speed)
    with torch.no_grad():
        for i, (test_imgs, test_clips_batch) in enumerate(test_loader):
            test_imgs = test_imgs.to(device)
            logits = trained_model(test_imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            clips.extend(test_clips_batch)

            # Break after 2 batches to keep demo fast
            if i >= 1:
                break

    # Create a dummy submission dataframe
    submission_df = pd.DataFrame({"clip": clips, "probability": predictions})

    print(f"Generated predictions for {len(submission_df)} test clips.")
    print(submission_df.head())

    # Verify probabilities are valid
    assert submission_df["probability"].min() >= 0.0, "Probabilities cannot be negative"
    assert submission_df["probability"].max() <= 1.0, "Probabilities cannot exceed 1.0"

    print("Inference assertions passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
