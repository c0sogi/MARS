import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, levenshtein_distance, compute_edit_distance_score
from library.data_loader import get_loaders
from library.model import SD_DGN
from library.losses import CascadedLoss
from library.trainer import Trainer
from library.inference import InferenceEngine, generate_submission


def run_demo():
    print("--- Starting SD-DGN Demo ---")

    # ==========================================
    # 1. Patch Configuration
    # ==========================================
    print("[1] Patching Configuration for Demo...")

    # Set working directory to a demo-specific path to avoid conflicts
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Optimize for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo script

    # Fix Class/Label Mismatch
    # Dataset contains labels 0-20 (21 classes), but default Config has 20.
    Config.NUM_CLASSES = 21

    # Create a mini test set for fast inference testing
    # (The provided inference script processes the whole file defined in Config)
    mini_test_path = os.path.join(Config.WORK_DIR, "mini_test.csv")
    if os.path.exists(Config.TEST_METADATA_PATH):
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        # Take a small subset
        df_test.head(Config.DEBUG_SUBSET_SIZE).to_csv(mini_test_path, index=False)
        Config.TEST_METADATA_PATH = mini_test_path
        print(f"    Created mini test metadata at {mini_test_path}")

    Config.print_config()

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[2] Verifying Utilities...")
    set_seed(42)

    # Test Levenshtein
    seq_a = [1, 2, 3]
    seq_b = [1, 2, 4]
    dist = levenshtein_distance(seq_a, seq_b)
    assert dist == 1, f"Levenshtein distance incorrect. Expected 1, got {dist}"

    # Test Score
    score = compute_edit_distance_score([seq_a], [seq_b])
    # Distance 1 / Length 3 = 0.333...
    assert abs(score - 1 / 3) < 1e-5, f"Score incorrect. Expected ~0.333, got {score}"
    print("    Utils verified.")

    # ==========================================
    # 3. Verify Data Loading
    # ==========================================
    print("\n[3] Verifying Data Loader...")
    # load_cached_data=False forces processing into our new demo cache dir
    loaders = get_loaders(load_cached_data=False)

    train_loader = loaders["train"]
    assert len(train_loader) > 0, "Train loader is empty."

    # Fetch a batch
    features, targets = next(iter(train_loader))

    # Check shapes
    # Features: [Batch, Channels, Time]
    # Targets: [Batch, Time]
    print(f"    Features Shape: {features.shape}")
    print(f"    Targets Shape: {targets.shape}")

    expected_channels = Config.INPUT_DIM_SKELETON + Config.INPUT_DIM_AUDIO
    assert (
        features.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Got {features.shape[0]}"
    assert (
        features.shape[1] == expected_channels
    ), f"Channel mismatch. Got {features.shape[1]}"
    assert (
        features.shape[2] == Config.WINDOW_SIZE
    ), f"Time dim mismatch. Got {features.shape[2]}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
    ), "Target shape mismatch."
    print("    Data Loader verified.")

    # ==========================================
    # 4. Verify Model
    # ==========================================
    print("\n[4] Verifying Model Architecture...")
    model = SD_DGN().to(Config.DEVICE)

    features = features.to(Config.DEVICE)
    outputs = model(features)

    # Model returns tuple: (logits1, logits2, logits3)
    assert isinstance(outputs, tuple), "Model output should be a tuple."
    assert len(outputs) == 3, "Model should return 3 outputs (multi-stage)."

    logits3 = outputs[2]
    print(f"    Logits Shape: {logits3.shape}")
    assert logits3.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        Config.WINDOW_SIZE,
    ), "Logits shape mismatch."
    print("    Model verified.")

    # ==========================================
    # 5. Verify Loss
    # ==========================================
    print("\n[5] Verifying Loss Function...")
    criterion = CascadedLoss().to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)

    loss, loss_dict = criterion(outputs, targets)
    print(f"    Loss Value: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."
    assert "loss" in loss_dict, "Loss dict missing total loss."
    print("    Loss verified.")

    # ==========================================
    # 6. Verify Trainer (Training Loop)
    # ==========================================
    print("\n[6] Verifying Trainer...")
    trainer = Trainer(loaders["train"], loaders["val"])

    # Run for 1 epoch
    trainer.fit(epochs=Config.EPOCHS)

    # Check if checkpoint was saved
    assert os.path.exists(
        trainer.checkpoint_path
    ), f"Checkpoint not found at {trainer.checkpoint_path}"
    print("    Trainer verified.")

    # ==========================================
    # 7. Verify Inference
    # ==========================================
    print("\n[7] Verifying Inference...")
    # Initialize engine (loads the checkpoint we just saved)
    engine = InferenceEngine(checkpoint_path=trainer.checkpoint_path)

    # Generate submission
    # load_cached_data=False forces processing of the mini test set
    generate_submission(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Check content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH, header=None)
    print(f"    Submission Rows: {len(df_sub)}")

    # We expect rows equal to the size of our mini test set (10)
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows, got {len(df_sub)}"

    print("    Inference verified.")

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
