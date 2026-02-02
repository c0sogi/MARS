import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, compute_auc
from library.dataset import get_dataloaders
from library.model import CoordinateAttentionCRNN
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration of Right Whale Detection Pipeline ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 40  # Small subset for quick processing
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use main thread to avoid overhead in demo

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Clean up previous working directory if it exists to ensure fresh run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    print(
        f"Configured: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # --------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading and Processing...")

    # This triggers audio processing (spectrogram generation)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch from training loader
    data_iter = iter(train_loader)
    specs, labels, ids = next(data_iter)

    print(f"Batch shapes -> Specs: {specs.shape}, Labels: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Freq, Time)
    # Freq = 128 (N_MELS)
    # Time ~ 126 (Duration * SampleRate / HopLength) -> 2.0 * 2000 / 32 = 125 + 1
    assert specs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert specs.shape[1] == 1, "Channel dimension should be 1"
    assert specs.shape[2] == Config.N_MELS, f"Freq dimension should be {Config.N_MELS}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"
    assert (
        isinstance(ids, tuple) or isinstance(ids, list) or isinstance(ids, np.ndarray)
    ), "IDs should be iterable"

    print("Data Loader verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    model = CoordinateAttentionCRNN().to(Config.DEVICE)
    model.eval()

    # Move batch to device
    specs_device = specs.to(Config.DEVICE)

    with torch.no_grad():
        output = model(specs_device)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output shape should be (Batch, 1)"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model verification passed.")

    # --------------------------------------------------------------------------
    # 4. Utility Logic Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Utility Functions (Mixup & AUC)...")

    # Test Mixup
    dummy_x = torch.randn(4, 1, 128, 128).to(Config.DEVICE)
    dummy_y = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(Config.DEVICE)

    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_x, dummy_y, alpha=1.0, device=Config.DEVICE
    )

    assert mixed_x.shape == dummy_x.shape, "Mixup altered input shape"
    assert y_a.shape == dummy_y.shape, "Mixup altered target shape"
    assert 0 <= lam <= 1, "Lambda should be between 0 and 1"

    # Test AUC
    true_labels = np.array([0, 0, 1, 1])
    pred_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc_score = compute_auc(true_labels, pred_scores)

    print(f"Computed AUC for dummy data: {auc_score}")
    assert 0 <= auc_score <= 1, "AUC must be between 0 and 1"

    print("Utility verification passed.")

    # --------------------------------------------------------------------------
    # 5. Trainer Execution (Train, Val, Predict)
    # --------------------------------------------------------------------------
    print("\n[Step 5] Running Trainer Loop (Train -> Val -> Predict)...")

    trainer = Trainer()

    # We use the loaders we already created
    # Run 1 Epoch of Training
    print("Training for 1 epoch...")
    train_loss = trainer.train_one_epoch(train_loader, epoch_idx=1)
    print(f"Train Loss: {train_loss:.4f}")

    # Run Validation
    print("Validating...")
    val_loss, val_auc = trainer.validate(val_loader)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save this model as 'best_model' manually for the prediction step
    # (Usually done by run() loop, but we are calling methods individually for demo)
    torch.save(trainer.model.state_dict(), Config.BEST_MODEL_PATH)

    # Run Prediction
    print("Predicting on Test Set...")
    trainer.predict(test_loader)

    print("Trainer execution complete.")

    # --------------------------------------------------------------------------
    # 6. Submission Verification
    # --------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Assertions
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Missing required columns"
    # In DEBUG mode, test loader size is limited by DEBUG_SAMPLES
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        df_sub["probability"].dtype == float
        or df_sub["probability"].dtype == np.float32
        or df_sub["probability"].dtype == np.float64
    ), "Probability column is not float"

    print("Submission verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
