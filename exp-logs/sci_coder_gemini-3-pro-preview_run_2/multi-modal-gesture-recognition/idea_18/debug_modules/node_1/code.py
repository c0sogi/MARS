import os
import shutil
import torch
import numpy as np
import random
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    levenshtein_distance,
    decode_predictions,
    compute_levenshtein_score,
)
from library.data_loader import get_dataloaders
from library.model import DSL_CRCN
from library.loss import DeepSupervisionLoss
from library.train import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    # 1. Setup & Configuration Overrides for Speed
    print(">>> Setting up configuration for demo run...")

    # Define temporary working directories
    demo_base_dir = "./working/demo_run"
    demo_cache_dir = os.path.join(demo_base_dir, "cache")
    demo_sub_dir = os.path.join(demo_base_dir, "submission")

    # Clean up previous run if exists
    if os.path.exists(demo_base_dir):
        shutil.rmtree(demo_base_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_sub_dir, exist_ok=True)

    # Override Config parameters
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_sub_dir
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for speed
    Config.BATCH_SIZE = 4
    Config.MAX_EPOCHS = 2
    Config.EARLY_STOPPING_PATIENCE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set device to CPU for consistent demo testing, or keep auto if GPU is desired
    # Config.DEVICE = 'cpu'

    set_seed(Config.SEED)
    print(f"    Cache Dir: {Config.CACHE_DIR}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # 2. Verify Utility Functions
    print("\n>>> Verifying Utility Functions...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]
    # Distance should be 1 (deletion of '2')
    dist = levenshtein_distance(seq1, seq2)
    assert dist == 1, f"Levenshtein distance incorrect. Expected 1, got {dist}"

    # Test Score Computation
    preds = [[1, 2], [3]]
    targets = [[1, 2], [3, 4]]
    # Dist 1: 0, Dist 2: 1. Total Dist: 1. Total Len: 2+2=4. Score: 0.25
    score = compute_levenshtein_score(preds, targets)
    assert score == 0.25, f"Levenshtein score incorrect. Expected 0.25, got {score}"

    # Test Decode Predictions
    # Mock probabilities: [Background, Class 1, Class 2]
    # Sequence: Class 1 -> Class 1 -> Background -> Class 2 -> Class 2
    # Should decode to [1, 2] after collapsing and removing background
    # Note: decode_predictions applies median filter.
    # Let's use a sequence long enough for kernel_size=3 (default in utils is 7, we'll override if possible or make seq long)
    # The provided utils.py has default kernel_size=7.
    T = 10
    num_classes = 3  # Background=0, 1, 2
    probs = np.zeros((T, num_classes))

    # Frames 0-3: Class 1
    probs[0:4, 1] = 1.0
    # Frames 4-5: Background
    probs[4:6, 0] = 1.0
    # Frames 6-9: Class 2
    probs[6:10, 2] = 1.0

    # With kernel size 3, the short background segment might be smoothed or kept depending on boundaries.
    # Let's just check it runs and returns a list.
    decoded = decode_predictions(probs, kernel_size=3)
    assert isinstance(decoded, list), "decode_predictions should return a list"
    assert all(
        isinstance(x, int) for x in decoded
    ), "decoded elements should be integers"
    print("    Utility functions verified.")

    # 3. Verify Data Loading
    print("\n>>> Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch
    features, targets, lengths = next(iter(train_loader))

    # Verify shapes
    # Features: (Batch, Time, D)
    # Targets: (Batch, Time)
    # Lengths: (Batch,)
    B, T, D = features.shape
    assert (
        B == Config.BATCH_SIZE or B == Config.DEBUG_SAMPLE_SIZE
    ), f"Batch size mismatch. Got {B}"
    assert (
        D == Config.INPUT_DIM
    ), f"Feature dimension mismatch. Expected {Config.INPUT_DIM}, got {D}"
    assert targets.shape == (
        B,
        T,
    ), f"Targets shape mismatch. Expected ({B}, {T}), got {targets.shape}"
    assert lengths.shape == (
        B,
    ), f"Lengths shape mismatch. Expected ({B},), got {lengths.shape}"

    print(
        f"    Batch Shapes - Features: {features.shape}, Targets: {targets.shape}, Lengths: {lengths.shape}"
    )
    print("    Data loading verified.")

    # 4. Verify Model Architecture
    print("\n>>> Verifying Model Architecture...")
    model = DSL_CRCN().to(Config.DEVICE)

    # Move batch to device
    features = features.to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)
    lengths = lengths.to(Config.DEVICE)

    # Create mask for forward pass (simulating what Trainer does)
    max_len = features.size(1)
    idx_range = torch.arange(max_len, device=Config.DEVICE).unsqueeze(0).expand(B, -1)
    mask = idx_range < lengths.unsqueeze(1)

    # Forward Pass
    outputs = model(features, mask=mask)
    stage1_out, stage2_out, stage3_out = outputs

    # Verify Output Shapes
    # Stage 1 & 2: (B, T, NumClasses + 1)
    # Stage 3: (B, T, NumClasses)
    expected_inter_dim = Config.NUM_CLASSES + Config.TRANSITION_CHANNELS

    assert stage1_out.shape == (
        B,
        T,
        expected_inter_dim,
    ), f"Stage 1 shape mismatch. Expected {(B, T, expected_inter_dim)}, got {stage1_out.shape}"
    assert stage2_out.shape == (
        B,
        T,
        expected_inter_dim,
    ), f"Stage 2 shape mismatch. Expected {(B, T, expected_inter_dim)}, got {stage2_out.shape}"
    assert stage3_out.shape == (
        B,
        T,
        Config.NUM_CLASSES,
    ), f"Stage 3 shape mismatch. Expected {(B, T, Config.NUM_CLASSES)}, got {stage3_out.shape}"

    print("    Model forward pass verified.")

    # 5. Verify Loss Function
    print("\n>>> Verifying Loss Function...")
    criterion = DeepSupervisionLoss().to(Config.DEVICE)

    loss, metrics = criterion(outputs, targets, lengths)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert "loss" in metrics, "Metrics dictionary missing 'loss'"
    assert "ce3" in metrics, "Metrics dictionary missing 'ce3'"

    print(f"    Computed Loss: {loss.item():.4f}")
    print("    Loss function verified.")

    # 6. Verify Full Training Loop
    print("\n>>> Verifying Trainer Integration (Running 2 Epochs)...")

    # Re-initialize trainer to ensure clean state
    trainer = Trainer()

    # Run fit
    trainer.fit()

    # Check if best model was saved
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"

    print("    Training loop execution successful.")
    print("    Best model saved.")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    try:
        run_demo()
    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        raise e
