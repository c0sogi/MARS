import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence
from library.data import MultiModalDataset, get_dataloaders
from library.models import AuxiliaryFusionNet
import library.train
import library.inference

# --- 1. Setup & Configuration ---
# Suppress warnings and progress bars for clean output
warnings.filterwarnings("ignore")


# Patch tqdm to disable progress bars in library modules
class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        if iterable is None:
            self.n = 0
            self.total = 0

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Apply the patch to the imported modules
library.train.tqdm = SilentTqdm
library.inference.tqdm = SilentTqdm


def main():
    print("--- Starting Task Execution ---")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Define working directories (Config uses ./working/idea_13 by default)
    print(f"Output Directory: {Config.OUTPUT_DIR}")

    # --- 2. Metric Verification ---
    print("\n[1/5] Verifying Metric (KL Divergence)...")
    # Case 1: Perfect prediction
    y_true = np.array([[0.2, 0.8]])
    y_pred_perf = np.array([[0.2, 0.8]])
    kl_perf = kl_divergence(y_true, y_pred_perf)
    assert np.isclose(
        kl_perf, 0.0, atol=1e-6
    ), f"Expected 0.0 KL for perfect pred, got {kl_perf}"

    # Case 2: Known divergence
    # P=[0.5, 0.5], Q=[0.25, 0.75]
    # KL = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75) = 0.5*ln(2) + 0.5*ln(2/3)
    #    = 0.5*0.6931 + 0.5*-0.4055 = 0.3465 - 0.2027 = 0.1438
    y_true_mix = np.array([[0.5, 0.5]])
    y_pred_mix = np.array([[0.25, 0.75]])
    kl_mix = kl_divergence(y_true_mix, y_pred_mix)
    assert kl_mix > 0.0, "KL should be positive for different distributions"
    print("Metric verification passed.")

    # --- 3. Data Loading Verification ---
    print("\n[2/5] Verifying Data Loading...")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Create dataset with small subset
    debug_size = 10
    subset_df = train_df.iloc[:debug_size]
    dataset = MultiModalDataset(subset_df, mode="train", augment=False)

    # Fetch one sample
    eeg_tensor, spec_tensor, target = dataset[0]

    # Check EEG shape: (Channels, Time) -> (20, 5000)
    assert eeg_tensor.shape == (
        Config.EEG_CHANNELS,
        Config.EEG_SEQ_LEN,
    ), f"Incorrect EEG shape: {eeg_tensor.shape}"

    # Check Spec shape: (Channels, H, W) -> (5, 512, 512)
    assert spec_tensor.shape == (
        Config.SPEC_CHANNELS,
        Config.SPEC_SIZE[0],
        Config.SPEC_SIZE[1],
    ), f"Incorrect Spec shape: {spec_tensor.shape}"

    # Check Target shape: (Num_Classes,) -> (6,)
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Incorrect Target shape: {target.shape}"

    print(
        f"Data shapes verified:\n - EEG: {eeg_tensor.shape}\n - Spec: {spec_tensor.shape}\n - Target: {target.shape}"
    )

    # --- 4. Model Architecture Verification ---
    print("\n[3/5] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuxiliaryFusionNet().to(device)

    # Create dummy batch (Batch Size = 2)
    dummy_eeg = eeg_tensor.unsqueeze(0).repeat(2, 1, 1).to(device)  # (2, 20, 5000)
    dummy_spec = (
        spec_tensor.unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    )  # (2, 5, 512, 512)

    model.eval()
    with torch.no_grad():
        joint, aux_eeg, aux_spec = model(dummy_eeg, dummy_spec)

    # Check output shapes: (Batch, Num_Classes)
    assert joint.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Joint output shape mismatch: {joint.shape}"
    assert aux_eeg.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Aux EEG output shape mismatch: {aux_eeg.shape}"
    assert aux_spec.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Aux Spec output shape mismatch: {aux_spec.shape}"

    print("Model forward pass successful. Output shapes verified.")

    # --- 5. Full Training Pipeline Demo ---
    print("\n[4/5] Running Training Pipeline (Debug Mode)...")
    # We use a small debug_limit to ensure this finishes quickly (e.g., 32 samples = 1 batch)
    # We run for 1 epoch.
    debug_limit = 32

    try:
        library.train.train(debug_limit=debug_limit, epochs=1)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # Verify outputs exist
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print(f"Training complete. Model saved to {Config.MODEL_PATH}")

    # --- 6. Inference Pipeline Verification ---
    print("\n[5/5] Running Inference Pipeline (Debug Mode)...")
    # Run inference explicitly using the saved model
    submission_df = library.inference.predict_and_submit(
        model_path=Config.MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
        metadata_path=Config.TEST_CSV,
        batch_size=4,  # Small batch for demo
        device=Config.DEVICE,
        debug_limit=10,  # Only predict 10 samples
    )

    # Verify submission structure
    expected_cols = ["eeg_id"] + Config.OUTPUT_COLS
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    assert (
        len(submission_df) == 10
    ), f"Expected 10 predictions, got {len(submission_df)}"

    # Verify probabilities sum to 1 (approx)
    prob_cols = Config.OUTPUT_COLS
    sums = submission_df[prob_cols].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    print("Inference successful. Submission file verified.")
    print("\n--- Task Completed Successfully ---")


if __name__ == "__main__":
    main()
