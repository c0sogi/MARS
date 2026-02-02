import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import PulmonaryDataset
from library.model import TriSlabModel
from library.train import run_training
from library.inference import generate_submission


def main():
    print("Starting Demonstration Script...")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")
    # Override Config values to run a fast integration test
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small test

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # ==========================================
    # 2. Verify Data Loading
    # ==========================================
    print("\n[2] Verifying Data Loading components...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = PulmonaryDataset(train_df, mode="train")
    print(f"Dataset length: {len(dataset)}")

    # Fetch one sample
    sample = dataset[0]
    img, tabular, base_fvc, time_delta, target = sample

    # Verify shapes and types
    print(f"Image shape: {img.shape}")
    print(f"Tabular shape: {tabular.shape}")

    # Assertions
    # Image should be (3, 224, 224) - 3 channels for Tri-Slab
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    # Tabular should be (4,) - Age, Percent, Sex, Smoking
    assert tabular.shape == (4,), f"Incorrect tabular shape: {tabular.shape}"
    # Target should be a scalar tensor
    assert isinstance(target, torch.Tensor), "Target is not a tensor"

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = TriSlabModel(Config).to(device)

    # Prepare batch
    img_batch = img.unsqueeze(0).to(device)  # (1, 3, 224, 224)
    tab_batch = tabular.unsqueeze(0).to(device)  # (1, 4)

    # Forward pass
    output = model(img_batch, tab_batch)
    print(f"Model output shape: {output.shape}")

    # Assertions
    # Output should be (Batch_Size, 3) -> [alpha, sigma_base, sigma_growth]
    assert output.shape == (1, 3), f"Expected output shape (1, 3), got {output.shape}"

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n[4] Verifying Loss Function...")

    loss_fn = LaplaceLogLikelihoodLoss()

    # Prepare inputs for loss
    targets_batch = target.unsqueeze(0).to(device)
    base_fvc_batch = base_fvc.unsqueeze(0).to(device)
    time_delta_batch = time_delta.unsqueeze(0).to(device)

    # Calculate loss
    loss = loss_fn(output, targets_batch, base_fvc_batch, time_delta_batch)
    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"

    # ==========================================
    # 5. Run Training Pipeline
    # ==========================================
    print("\n[5] Running Training Pipeline (Integration Test)...")

    # This function handles the training loop, validation, and model saving
    # We run in debug mode to use the small subset defined in Step 1
    run_training(debug=True)

    # Verify model was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved!"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 6. Run Inference Pipeline
    # ==========================================
    print("\n[6] Running Inference Pipeline...")

    # This function loads the saved model and generates submission.csv
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated!"

    # Load and validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), f"Missing columns. Expected {expected_cols}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaN values"

    # Check FVC and Confidence values are reasonable
    # FVC should be positive, Confidence should be >= 70 (clipped)
    assert (sub_df["FVC"] > 0).all(), "Negative FVC predictions found"
    assert (sub_df["Confidence"] >= 70).all(), "Confidence values below 70 found"

    print("\nSuccess! All components verified and pipeline executed.")


if __name__ == "__main__":
    main()
