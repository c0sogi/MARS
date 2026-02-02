import os
import sys
import shutil
import torch
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ==========================================
# 1. Configuration Override for Speed & Demo
# ==========================================
# We import config first and modify it before other modules import from it.
import library.config

# Set a small subset for rapid execution
library.config.DEBUG_SUBSET_SIZE = 20
# Minimize training duration
library.config.NUM_EPOCHS = 2
library.config.BATCH_SIZE = 4
library.config.NUM_WORKERS = (
    0  # Use main process to avoid multiprocessing overhead in demo
)
# Use a custom working directory for this demo to ensure a clean state
demo_working_dir = "./working/demo_execution"
library.config.WORKING_DIR = demo_working_dir
library.config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")

# Ensure directories exist (since we changed the path after config's initial run)
os.makedirs(library.config.WORKING_DIR, exist_ok=True)
os.makedirs(library.config.CHECKPOINT_DIR, exist_ok=True)

# Clean up previous stats if they exist to force re-computation on the subset
stats_path = os.path.join(demo_working_dir, "stats.npz")
if os.path.exists(stats_path):
    os.remove(stats_path)

# ==========================================
# 2. Import Library Modules
# ==========================================
# Now it is safe to import modules that depend on the config
from library.utils import set_seed, levenshtein_distance, decode_predictions
from library.data_loader import get_loaders, ItalianGestureDataset
from library.model import GCINet
from library.train import train_model


def test_utils():
    """Verifies utility functions."""
    print("\n[1/4] Testing Utilities...")

    # Test Levenshtein Distance
    # Case 1: Identical sequences
    dist_eq = levenshtein_distance([1, 2, 3], [1, 2, 3])
    assert dist_eq == 0, f"Levenshtein: Expected 0, got {dist_eq}"

    # Case 2: One insertion/deletion/substitution
    dist_diff = levenshtein_distance([1, 2, 3], [1, 2])
    assert dist_diff == 1, f"Levenshtein: Expected 1, got {dist_diff}"

    # Test Decode Predictions
    # Create synthetic logits (Time=20, Classes=21)
    # Class 1 for first 10 frames, Class 2 for next 10 frames
    # Note: decode_predictions filters segments < 5 frames
    logits = torch.zeros(20, library.config.MODEL_OUTPUT_CLASSES)
    # High logit for Class 1 (index 1)
    logits[0:10, 1] = 10.0
    # High logit for Class 2 (index 2)
    logits[10:20, 2] = 10.0

    decoded = decode_predictions(logits)
    expected = [1, 2]
    assert decoded == expected, f"Decode: Expected {expected}, got {decoded}"

    print("Utilities verified.")


def test_data_pipeline():
    """Verifies Dataset and DataLoader."""
    print("\n[2/4] Testing Data Pipeline...")

    # Initialize Loaders
    # This will trigger statistics computation on the 20-sample subset
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=library.config.BATCH_SIZE, num_workers=library.config.NUM_WORKERS
    )

    assert len(train_loader) > 0, "Train loader is empty."

    # Fetch one batch
    batch = next(iter(train_loader))

    # Unpack
    skel = batch["skeleton"]
    audio = batch["audio"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    mask = batch["mask"]

    # Verify Shapes
    B = library.config.BATCH_SIZE
    # Skeleton: (B, T, 60)
    assert (
        skel.dim() == 3 and skel.shape[0] == B and skel.shape[2] == 60
    ), f"Skeleton shape mismatch: {skel.shape}"

    # Audio: (B, T, 64)
    assert (
        audio.dim() == 3 and audio.shape[0] == B and audio.shape[2] == 64
    ), f"Audio shape mismatch: {audio.shape}"

    # Labels: (B, T)
    assert (
        labels.dim() == 2 and labels.shape[0] == B
    ), f"Labels shape mismatch: {labels.shape}"

    print(f"Batch verified. Skeleton: {skel.shape}, Audio: {audio.shape}")
    return train_loader


def test_model_forward(loader):
    """Verifies Model Architecture and Forward Pass."""
    print("\n[3/4] Testing Model Forward Pass...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCINet().to(device)

    # Get a batch
    batch = next(iter(loader))
    skel = batch["skeleton"].to(device)
    audio = batch["audio"].to(device)
    lengths = batch["lengths"].to(device)

    # Forward Pass
    logits = model(skel, audio, lengths)

    # Verify Output Shape: (B, T, NumClasses+1)
    expected_classes = library.config.MODEL_OUTPUT_CLASSES
    assert logits.dim() == 3, "Logits should be 3D"
    assert logits.shape[0] == skel.shape[0], "Batch size mismatch"
    assert logits.shape[1] == skel.shape[1], "Temporal dimension mismatch"
    assert (
        logits.shape[2] == expected_classes
    ), f"Class dim mismatch. Expected {expected_classes}, got {logits.shape[2]}"

    print(f"Model forward pass successful. Logits shape: {logits.shape}")


def test_training_integration():
    """Verifies the full training loop."""
    print("\n[4/4] Testing Training Loop Integration...")

    # Run training with the modified config (2 epochs, subset of 20)
    # This tests optimizer, loss, validation, and saving.
    best_model_path = train_model(
        num_epochs=library.config.NUM_EPOCHS, batch_size=library.config.BATCH_SIZE
    )

    # Verify artifact creation
    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"

    print(f"Training integration successful. Model saved to: {best_model_path}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    print("Starting Demonstration Script...")
    print(
        f"Config: Subset={library.config.DEBUG_SUBSET_SIZE}, Epochs={library.config.NUM_EPOCHS}, Batch={library.config.BATCH_SIZE}"
    )

    try:
        # 1. Utils
        test_utils()

        # 2. Data
        loader = test_data_pipeline()

        # 3. Model
        test_model_forward(loader)

        # 4. Training
        test_training_integration()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nRUNTIME ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
