import os
import shutil
import warnings
import torch
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import the Config class
from library.config import Config

print("[Demo] Setting up configuration for lightweight execution...")

# Patch Config for a fast demonstration run
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 50  # Process only 50 rows for speed
Config.EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 8  # Small batch size
Config.NUM_WORKERS = 2  # Minimize multiprocessing overhead
Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety
Config.WORKING_DIR = "./working/demo_execution"
Config.CACHE_DIR = Config.WORKING_DIR
Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Clean and recreate the working directory for this demo
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import library modules after patching Config
from library.utils import seed_everything, KLDivLossWithLogits
from library.data_loader import get_dataloaders
from library.models import DualStreamNetwork
from library.train import run_training
from library.inference import predict_test_set

if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[Demo] Verifying Data Pipeline...")

    # Generate DataLoaders (this will trigger data processing and caching)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch to verify shapes and types
    eeg_batch, spec_batch, target_batch = next(iter(train_loader))

    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  EEG Shape: {eeg_batch.shape} (Expected: [{Config.BATCH_SIZE}, 2500, 19])")
    print(
        f"  Spec Shape: {spec_batch.shape} (Expected: [{Config.BATCH_SIZE}, 4, 256, 256])"
    )
    print(f"  Target Shape: {target_batch.shape} (Expected: [{Config.BATCH_SIZE}, 6])")

    # Assertions
    assert eeg_batch.shape == (Config.BATCH_SIZE, 2500, 19), "Incorrect EEG batch shape"
    assert spec_batch.shape == (
        Config.BATCH_SIZE,
        4,
        256,
        256,
    ), "Incorrect Spectrogram batch shape"
    assert target_batch.shape == (Config.BATCH_SIZE, 6), "Incorrect Target batch shape"
    assert not torch.isnan(eeg_batch).any(), "EEG batch contains NaNs"
    assert not torch.isnan(spec_batch).any(), "Spectrogram batch contains NaNs"

    print("  Data Pipeline verification passed.")

    # ==========================================
    # 3. Model & Loss Verification
    # ==========================================
    print("\n[Demo] Verifying Model Architecture and Loss...")

    device = torch.device(Config.DEVICE)
    model = DualStreamNetwork(
        num_classes=Config.N_CLASSES, pretrained=Config.PRETRAINED
    )
    model.to(device)

    # Move batch to device
    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass
    logits = model(eeg_batch, spec_batch)

    print(f"  Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 6), "Model output shape mismatch"

    # Loss computation
    criterion = KLDivLossWithLogits()
    loss = criterion(logits, target_batch)

    print(f"  Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("  Model and Loss verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[Demo] Executing Training Loop (1 Epoch)...")

    # Run the full training orchestration
    # This uses the patched Config (1 epoch, debug data)
    run_training()

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print(f"  Training complete. Model saved to {Config.MODEL_PATH}")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    print("\n[Demo] Executing Inference on Test Set...")

    # Run inference
    predict_test_set(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    # Load and validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Rows: {len(df_sub)}")
    print("  First 3 rows:")
    print(df_sub.head(3))

    # Check columns
    expected_cols = ["eeg_id"] + Config.SUBMISSION_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check probability sum (should be approx 1.0)
    prob_sums = df_sub[Config.SUBMISSION_COLS].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("  Inference verification passed.")
    print("\n[Demo] All demonstration steps completed successfully.")
