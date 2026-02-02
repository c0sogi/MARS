import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.data import BrainDataset, get_dataloaders
from library.models import BidirectionalFusionNet
from library.utils import seed_everything, kl_divergence_score
from library.train import train_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast execution...")

    # Modify Config global state
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_CSV = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist (Config.setup() does this, but we do it here for safety before manual steps)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set Seed
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n2. Verifying Data Pipeline...")

    # Load metadata manually to test Dataset class
    train_df = pd.read_csv(Config.TRAIN_CSV).iloc[: Config.DEBUG_SUBSET_SIZE]

    # Initialize Dataset
    ds = BrainDataset(train_df, Config, mode="train", augment=False)

    # Check length
    assert (
        len(ds) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(ds)}"

    # Fetch one sample
    sample = ds[0]

    # Verify Keys
    expected_keys = {"eeg", "spec", "target"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset output. Found: {sample.keys()}"

    # Verify Shapes
    # EEG: (Channels, Time) -> (20, 5000)
    assert sample["eeg"].shape == (
        Config.EEG_CHANNELS,
        Config.EEG_SEQ_LEN,
    ), f"EEG shape mismatch. Expected {(Config.EEG_CHANNELS, Config.EEG_SEQ_LEN)}, got {sample['eeg'].shape}"

    # Spec: (Channels, H, W) -> (5, 512, 512)
    assert sample["spec"].shape == (
        Config.SPEC_CHANNELS,
        Config.SPEC_IMG_SIZE[0],
        Config.SPEC_IMG_SIZE[1],
    ), f"Spectrogram shape mismatch. Expected {(Config.SPEC_CHANNELS, *Config.SPEC_IMG_SIZE)}, got {sample['spec'].shape}"

    # Target: (Num_Classes,) -> (6,)
    assert sample["target"].shape == (
        Config.NUM_CLASSES,
    ), f"Target shape mismatch. Expected {(Config.NUM_CLASSES,)}, got {sample['target'].shape}"

    print("   -> Dataset verification passed.")

    # Verify DataLoader
    train_loader, _, _ = get_dataloaders(Config)
    batch = next(iter(train_loader))

    assert batch["eeg"].shape == (
        Config.BATCH_SIZE,
        Config.EEG_CHANNELS,
        Config.EEG_SEQ_LEN,
    ), "Batch EEG shape mismatch"
    assert batch["spec"].shape == (
        Config.BATCH_SIZE,
        Config.SPEC_CHANNELS,
        *Config.SPEC_IMG_SIZE,
    ), "Batch Spec shape mismatch"
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Batch Target shape mismatch"

    print("   -> DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Logic
    # -------------------------------------------------------------------------
    print("\n3. Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = BidirectionalFusionNet(Config).to(device)
    model.eval()

    # Create dummy input based on config
    dummy_eeg = torch.randn(2, Config.EEG_CHANNELS, Config.EEG_SEQ_LEN).to(device)
    dummy_spec = torch.randn(2, Config.SPEC_CHANNELS, *Config.SPEC_IMG_SIZE).to(device)

    with torch.no_grad():
        logits = model(dummy_eeg, dummy_spec)

    # Verify Output Shape: (Batch, Num_Classes)
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {logits.shape}"

    # Verify no NaNs
    assert not torch.isnan(logits).any(), "Model produced NaN logits."

    print("   -> Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Metric Logic
    # -------------------------------------------------------------------------
    print("\n4. Verifying Metric (KL Divergence)...")

    # Case 1: Identical distributions (KL should be 0)
    y_true = np.array([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])
    y_pred = np.array([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])
    score = kl_divergence_score(y_true, y_pred)
    assert np.isclose(
        score, 0.0, atol=1e-6
    ), f"Metric failed for identical inputs. Got {score}"

    # Case 2: Different distributions
    y_pred_diff = np.array([[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]])
    score_diff = kl_divergence_score(y_true, y_pred_diff)
    assert score_diff > 0, "Metric should be positive for different distributions"

    print("   -> Metric verification passed.")

    # -------------------------------------------------------------------------
    # 5. Full Integration Run (Train Loop)
    # -------------------------------------------------------------------------
    print("\n5. Running Full Training Integration Test...")
    print("   (This runs train_model() with DEBUG=True, 1 Epoch, 20 samples)")

    # Execute the training pipeline provided in library/train.py
    # This will use the modified Config settings
    train_model()

    # -------------------------------------------------------------------------
    # 6. Verify Submission Output
    # -------------------------------------------------------------------------
    print("\n6. Verifying Output Artifacts...")

    if os.path.exists(Config.SUBMISSION_CSV):
        sub_df = pd.read_csv(Config.SUBMISSION_CSV)
        print(f"   -> Submission file found with {len(sub_df)} rows.")

        # Check columns
        expected_cols = ["eeg_id"] + [f"{c}_vote" for c in Config.CLASS_NAMES]
        assert all(
            col in sub_df.columns for col in expected_cols
        ), "Submission columns mismatch"

        # Check probability sum
        prob_cols = [c for c in sub_df.columns if "_vote" in c]
        sums = sub_df[prob_cols].sum(axis=1)
        assert np.allclose(
            sums, 1.0, atol=1e-4
        ), "Submission probabilities do not sum to 1.0"

        print("   -> Submission format verification passed.")
    else:
        # If test data wasn't available (e.g., empty test.csv), train_model might skip submission
        # Check if test.csv exists in metadata
        if os.path.exists(Config.TEST_CSV):
            raise FileNotFoundError(
                f"Submission file was not created at {Config.SUBMISSION_CSV}"
            )
        else:
            print(
                "   -> No test metadata found, submission generation correctly skipped."
            )

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
