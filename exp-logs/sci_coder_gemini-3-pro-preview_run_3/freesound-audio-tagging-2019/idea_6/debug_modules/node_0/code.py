import os
import sys
import shutil
import torch
import numpy as np
import warnings

# Import library modules
from library.configuration import Config
from library.utilities import set_seed, calculate_lrap, mixup_data, mixup_criterion
from library.data_loader import get_dataloaders, get_test_dataloader
from library.network import ConvNeXtAudio
from library.trainer import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_main():
    print(">>> Starting Audio Tagging Library Demo")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("\n[1] Setting up Demo Configuration...")

    class DemoConfig(Config):
        """
        Modified configuration for a quick demonstration run.
        """

        # Use a separate output directory for the demo to avoid conflicts
        IDEA_NAME = "demo_run"
        OUTPUT_ROOT = os.path.join("./working", IDEA_NAME)

        # Ensure output directory exists
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        # Paths
        BEST_MODEL_PATH = os.path.join(OUTPUT_ROOT, "best_model.pth")
        SUBMISSION_PATH = os.path.join(OUTPUT_ROOT, "submission.csv")

        # Speed optimizations
        DEBUG = True
        DEBUG_SUBSET_SIZE = 100  # Process only 100 samples
        EPOCHS = 1  # Train for only 1 epoch
        BATCH_SIZE = 8  # Small batch size
        NUM_WORKERS = 2  # Reduce workers overhead

        # Model params (keep same as original for validity, but can adjust if needed)
        BACKBONE = "convnext_nano"

    # Print config to verify
    DemoConfig.print_config()

    # Set seed for reproducibility
    set_seed(DemoConfig.SEED)
    print("Seed set successfully.")

    # ------------------------------------------------------------------------
    # 2. Data Loader Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Force load_cached_data=False to ensure we generate the debug subset fresh
    train_loader, val_loader = get_dataloaders(DemoConfig, load_cached_data=False)

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # Verify Train Batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Freq, Time) -> (8, 1, 128, T)
    # Time dimension depends on DURATION (30s) and HOP_LENGTH.
    # 30s * 32000sr / 512hop ~= 1875 frames.
    expected_frames = (
        int(DemoConfig.SAMPLE_RATE * DemoConfig.DURATION / DemoConfig.HOP_LENGTH) + 1
    )
    # Allow small padding differences
    assert images.shape[0] == DemoConfig.BATCH_SIZE, "Incorrect batch size"
    assert images.shape[1] == 1, "Input should be 1 channel (mono)"
    assert images.shape[2] == DemoConfig.N_MELS, "Incorrect Mel bins"
    assert (
        abs(images.shape[3] - expected_frames) < 10
    ), f"Unexpected time dim: {images.shape[3]}"
    assert labels.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.NUM_CLASSES,
    ), "Incorrect label shape"

    # Verify Data Statistics (Spectrograms should not be empty/silent unless corrupted)
    assert torch.mean(images) != 0, "Images appear to be empty (all zeros)"
    print("Data Loader verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = ConvNeXtAudio(config=DemoConfig)
    model = model.to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.NUM_CLASSES,
    ), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"
    print("Model verification passed.")

    # ------------------------------------------------------------------------
    # 4. Utilities Verification (Mixup & Metric)
    # ------------------------------------------------------------------------
    print("\n[4] Testing Utilities...")

    # Test Mixup
    mixed_x, y_a, y_b, lam = mixup_data(
        images, labels.to(device), alpha=0.4, device=device
    )
    assert mixed_x.shape == images.shape, "Mixup altered input shape"
    assert 0 <= lam <= 1, "Lambda out of range"
    print("Mixup verification passed.")

    # Test LRAP Metric
    # Create synthetic ground truth and predictions
    # Case 1: Perfect prediction
    y_true_mock = np.array([[1, 0, 1], [0, 1, 0]])
    y_score_mock = np.array(
        [[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]]
    )  # High scores for true classes

    score_perfect = calculate_lrap(y_true_mock, y_score_mock)
    print(f"Mock LRAP (Good Predictions): {score_perfect:.4f}")
    assert score_perfect == 1.0, "Perfect predictions should yield LRAP 1.0"

    # Case 2: Bad prediction
    y_score_bad = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.8]])
    score_bad = calculate_lrap(y_true_mock, y_score_bad)
    print(f"Mock LRAP (Bad Predictions): {score_bad:.4f}")
    assert score_bad < 1.0, "Bad predictions should yield LRAP < 1.0"

    print("Metric verification passed.")

    # ------------------------------------------------------------------------
    # 5. Full Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Demo Mode)...")

    # We use the run_training function from trainer.py
    # This handles the loop, validation, and saving.
    # We pass load_cached_data=True now, as we generated the cache in step [2]
    # (Step 2 called get_dataloaders which saves to cache).

    best_score = run_training(config=DemoConfig, load_cached_data=True)

    print(f"Training finished. Best Validation Score: {best_score:.4f}")

    # Verify artifacts
    assert os.path.exists(DemoConfig.BEST_MODEL_PATH), "Best model file was not saved"
    print(f"Model saved at: {DemoConfig.BEST_MODEL_PATH}")

    # ------------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n[6] Testing Inference DataLoader...")

    test_loader = get_test_dataloader(DemoConfig, load_cached_data=False)
    test_images, _ = next(iter(test_loader))

    assert test_images.shape[0] == DemoConfig.BATCH_SIZE, "Test batch size mismatch"
    print(f"Test batch shape: {test_images.shape}")
    print("Inference setup verification passed.")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    demo_main()
