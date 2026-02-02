import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from provided library files
from library.config import Config
from library.data_transforms import MultiResolutionSTFT, get_transforms
from library.dataset import EEGDataset
from library.model import MultiResDualStreamNet
from library.trainer import Trainer
from library.inference import predict
from library.utils import seed_everything


# ==========================================
# 1. Configuration Setup for Demo
# ==========================================
class DemoConfig(Config):
    """
    Configuration override for a fast demonstration run.
    """

    # Execution Control
    EPOCHS = 1
    BATCH_SIZE = 4
    DEBUG = True
    DEBUG_SUBSET_SIZE = 16  # Small subset for speed
    NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Paths
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = WORKING_DIR

    # Ensure clean start
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)


def run_demo():
    print("Starting End-to-End Demonstration...")
    seed_everything(DemoConfig.SEED)

    # ==========================================
    # 2. Verify Data Transforms (Stream A)
    # ==========================================
    print("\n[1/5] Verifying Data Transforms...")

    # Create synthetic EEG data: (Time, Channels) -> (10000, 19)
    synthetic_eeg = np.random.randn(
        DemoConfig.TOTAL_SAMPLES, DemoConfig.N_EEG_CHANNELS
    ).astype(np.float32)

    # Initialize STFT
    stft_transform = MultiResolutionSTFT()

    # Apply STFT
    # Expected output: (Freq, Time, Channels*Resolutions) -> (128, 500, 57)
    stft_output = stft_transform(synthetic_eeg)
    print(f"  STFT Output Shape: {stft_output.shape}")

    assert stft_output.shape == (
        128,
        500,
        57,
    ), f"STFT output shape mismatch. Expected (128, 500, 57), got {stft_output.shape}"

    # Apply Albumentations Pipeline
    transforms = get_transforms(mode="train", data_type="eeg")
    aug_output = transforms(image=stft_output)["image"]

    # Expected tensor output: (Channels, Freq, Time) -> (57, 128, 500)
    print(f"  Augmented Tensor Shape: {aug_output.shape}")
    assert aug_output.shape == (
        57,
        128,
        500,
    ), f"Augmented tensor shape mismatch. Expected (57, 128, 500), got {aug_output.shape}"

    print("  Transforms verified successfully.")

    # ==========================================
    # 3. Verify Dataset Loading
    # ==========================================
    print("\n[2/5] Verifying Dataset...")

    # Initialize Dataset with DemoConfig (uses subset)
    dataset = EEGDataset(
        mode="train",
        config=DemoConfig,
        load_cached_data=False,  # Force processing for demo
        subset_size=DemoConfig.DEBUG_SUBSET_SIZE,
    )

    print(f"  Dataset Length: {len(dataset)}")
    assert (
        len(dataset) == DemoConfig.DEBUG_SUBSET_SIZE
    ), "Dataset subset size incorrect."

    # Fetch one sample
    (x_a, x_b), target = dataset[0]

    # Verify Stream A (EEG)
    # Shape: (57, 128, 500)
    assert x_a.shape == (57, 128, 500), f"Stream A shape incorrect: {x_a.shape}"

    # Verify Stream B (Spectrogram)
    # Shape: (4, 256, 256)
    assert x_b.shape == (4, 256, 256), f"Stream B shape incorrect: {x_b.shape}"

    # Verify Target
    # Shape: (6,)
    assert target.shape == (6,), f"Target shape incorrect: {target.shape}"

    print("  Dataset shapes verified successfully.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[3/5] Verifying Model...")

    device = torch.device(DemoConfig.DEVICE)
    model = MultiResDualStreamNet(
        pretrained=False
    )  # Skip download for speed/offline safety
    model.to(device)
    model.eval()

    # Create a dummy batch
    # Stream A: (Batch, 57, 128, 500)
    dummy_a = torch.randn(2, 57, 128, 500).to(device)
    # Stream B: (Batch, 4, 256, 256)
    dummy_b = torch.randn(2, 4, 256, 256).to(device)

    with torch.no_grad():
        logits = model((dummy_a, dummy_b))

    print(f"  Model Output Shape: {logits.shape}")
    assert logits.shape == (
        2,
        6,
    ), f"Model output shape mismatch. Expected (2, 6), got {logits.shape}"

    print("  Model forward pass verified successfully.")

    # ==========================================
    # 5. Run Training Loop (Trainer)
    # ==========================================
    print("\n[4/5] Running Training Loop...")

    # Initialize Trainer with DemoConfig
    trainer = Trainer(config=DemoConfig, debug=True)

    # Run Fit (1 Epoch)
    best_loss = trainer.fit()

    # Verify Checkpoint
    checkpoint_path = os.path.join(DemoConfig.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not created."
    print(f"  Training completed. Best Loss: {best_loss:.4f}")
    print(f"  Checkpoint saved at: {checkpoint_path}")

    # ==========================================
    # 6. Run Inference
    # ==========================================
    print("\n[5/5] Running Inference...")

    submission_path = os.path.join(DemoConfig.WORKING_DIR, "submission.csv")

    # Run prediction
    # Note: We use the checkpoint generated in the previous step
    submission = predict(
        config=DemoConfig,
        checkpoint_path=checkpoint_path,
        output_path=submission_path,
        batch_size=DemoConfig.BATCH_SIZE,
        num_workers=0,
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file not found."
    assert len(submission) > 0, "Submission file is empty."
    assert all(
        col in submission.columns for col in DemoConfig.CLASS_NAMES
    ), "Missing columns in submission."

    # Check probability sum constraint (should be approx 1.0)
    prob_sums = submission[DemoConfig.CLASS_NAMES].sum(axis=1)
    # Allow small float error
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("  Inference completed successfully.")
    print(f"  Submission Head:\n{submission.head(2)}")

    print("\nAll demonstration steps passed successfully!")


if __name__ == "__main__":
    run_demo()
