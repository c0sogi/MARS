import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import TrainConfig, AudioConfig, ModelConfig
from library.dataset import get_dataloaders, WhaleDataset
from library.model import WhaleDetector
from library.trainer import Trainer
from library.utils import set_seed


def main():
    print("=== Starting Demonstration Script ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demonstration
    # ---------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Modify TrainConfig to run a quick debug session
    TrainConfig.debug = True
    TrainConfig.debug_samples = 20  # Use only 20 samples per split
    TrainConfig.epochs = 1  # Train for only 1 epoch
    TrainConfig.batch_size = 4  # Small batch size
    TrainConfig.num_workers = 0  # Avoid multiprocessing overhead for small data

    # Redirect outputs to a demo specific directory
    DEMO_DIR = "./working/demo_execution"
    TrainConfig.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    TrainConfig.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    TrainConfig.SUBMISSION_PATH = os.path.join(
        TrainConfig.SUBMISSION_DIR, "submission.csv"
    )
    TrainConfig.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Clean up previous demo runs if any
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(TrainConfig.CACHE_DIR, exist_ok=True)
    os.makedirs(TrainConfig.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(TrainConfig.seed)
    print("   Configuration updated successfully.")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n2. Verifying Data Pipeline...")

    # Initialize DataLoaders
    # We force re-computation (load_cached_data=False) to test the spectrogram generation logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    print("   Checking Train Loader...")
    images, labels = next(iter(train_loader))

    # Expected dimensions
    # Batch Size: 4
    # Channels: 1
    # Freq (Mel bins): 320 (from AudioConfig.n_mels)
    # Time: ~201 (Sample Rate 2000 * Duration 2.0 / Hop Length 20 + 1)
    expected_time_dim = (
        int(AudioConfig.sr * AudioConfig.duration / AudioConfig.hop_length) + 1
    )

    assert (
        images.shape[0] == TrainConfig.batch_size
    ), f"Batch size mismatch: {images.shape[0]}"
    assert images.shape[1] == 1, f"Channel mismatch: {images.shape[1]}"
    assert (
        images.shape[2] == AudioConfig.n_mels
    ), f"Mel bins mismatch: {images.shape[2]}"
    # Allow small tolerance in time dimension due to padding/centering
    assert (
        abs(images.shape[3] - expected_time_dim) <= 2
    ), f"Time dim mismatch: {images.shape[3]} vs {expected_time_dim}"
    assert labels.shape == (
        TrainConfig.batch_size,
    ), f"Label shape mismatch: {labels.shape}"

    print(f"   Train batch shape verified: {images.shape}")
    print(f"   Train label shape verified: {labels.shape}")

    # Verify Test Loader (No labels)
    print("   Checking Test Loader...")
    test_images = next(iter(test_loader))
    assert test_images.shape[0] == TrainConfig.batch_size, "Test batch size mismatch"
    assert isinstance(test_loader.dataset, WhaleDataset), "Test dataset type mismatch"
    print("   Test batch shape verified.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n3. Verifying Model Architecture...")

    model = WhaleDetector()
    model.eval()  # Set to eval mode

    # Create a dummy input tensor matching the data shape
    dummy_input = torch.randn(2, 1, AudioConfig.n_mels, images.shape[3])

    print(f"   Forward pass with input: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch_Size, Num_Classes) -> (2, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print(f"   Output shape verified: {output.shape}")

    # Check internal components
    print(f"   Backbone: {ModelConfig.model_name}")
    print(f"   Pooling: {'GeM' if ModelConfig.use_gem else 'AvgPool'}")
    print(f"   Attention: {'CoordAtt' if ModelConfig.use_coord_att else 'None'}")

    # ---------------------------------------------------------
    # 4. Trainer and Training Loop Verification
    # ---------------------------------------------------------
    print("\n4. Verifying Trainer and Training Loop...")

    # Initialize Trainer
    # It will reload dataloaders, this time using the cache we just generated in step 2
    trainer = Trainer(load_cached_data=True)

    # Run Training (Fit)
    print("   Starting training (1 epoch)...")
    trainer.fit(epochs=TrainConfig.epochs)

    # Verify Checkpoint creation
    assert os.path.exists(
        TrainConfig.CHECKPOINT_PATH
    ), "Checkpoint file was not created."
    print(f"   Checkpoint verified at {TrainConfig.CHECKPOINT_PATH}")

    # Run Inference (Predict)
    print("   Generating predictions...")
    trainer.predict()

    # Verify Submission file
    assert os.path.exists(
        TrainConfig.SUBMISSION_PATH
    ), "Submission file was not created."

    df_sub = pd.read_csv(TrainConfig.SUBMISSION_PATH)
    print(f"   Submission loaded. Shape: {df_sub.shape}")

    # Verify Submission Content
    assert "clip" in df_sub.columns, "Submission missing 'clip' column"
    assert "probability" in df_sub.columns, "Submission missing 'probability' column"
    assert (
        len(df_sub) == TrainConfig.debug_samples
    ), f"Submission row count mismatch. Expected {TrainConfig.debug_samples}, got {len(df_sub)}"

    # Check probability range
    probs = df_sub["probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("   Submission format and content verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
