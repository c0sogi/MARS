import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import library modules
from library.config import Config
from library.utils import set_seed, compute_levenshtein_ratio, decode_predictions
from library.data_loader import get_dataloaders, SkeletonAudioDataset
from library.model import MVAIIN
from library.trainer import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a temporary environment with a subset of data to ensure
    the demonstration runs quickly and respects the read-only input directory.
    """
    print("Setting up demo environment...")

    # Define paths
    demo_work_dir = "./working/demo_execution"
    demo_metadata_dir = os.path.join(demo_work_dir, "metadata")

    # Clean up previous run if exists
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    os.makedirs(demo_metadata_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Create subsets (Top 10 samples for speed)
    subset_size = 10
    train_mini = train_df.head(subset_size)
    val_mini = val_df.head(subset_size)
    test_mini = test_df.head(subset_size)

    # Save mini metadata
    train_mini.to_csv(os.path.join(demo_metadata_dir, "train.csv"), index=False)
    val_mini.to_csv(os.path.join(demo_metadata_dir, "val.csv"), index=False)
    test_mini.to_csv(os.path.join(demo_metadata_dir, "test.csv"), index=False)

    print(f"Created mini datasets with {subset_size} samples each.")

    # Override Config to use demo paths and settings
    Config.METADATA_DIR = demo_metadata_dir
    Config.WORK_DIR = demo_work_dir
    Config.CACHE_DIR = os.path.join(demo_work_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_work_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Training Hyperparameters for Speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 2

    print("Configuration updated for demo execution.")


def validate_utils():
    """
    Validates utility functions logic.
    """
    print("\nValidating Utils...")

    # Test Levenshtein Ratio
    # Case 1: Perfect match
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    score = compute_levenshtein_ratio([seq1], [seq2])
    assert score == 0.0, f"Expected 0.0 error for perfect match, got {score}"

    # Case 2: Complete mismatch
    seq3 = [1]
    seq4 = [2]
    score = compute_levenshtein_ratio([seq3], [seq4])
    # Distance is 1, length is 1 -> ratio 1.0
    assert score == 1.0, f"Expected 1.0 error for mismatch, got {score}"

    # Test Decode Predictions
    # Config.MIN_SEGMENT_LENGTH is default 5. Config.BACKGROUND_CLASS_ID is 0.
    # Create a sequence: Background(5) -> Class1(5) -> Background(2) -> Class2(4) -> Class2(6)
    # Class2(4) is too short (if treated as separate), but here we simulate raw frames.
    # Let's try: 0,0,0,0,0, 1,1,1,1,1, 0,0, 2,2,2,2,2,2
    raw_preds = [0] * 5 + [1] * 5 + [0] * 2 + [2] * 6
    decoded = decode_predictions(raw_preds, min_segment_length=5, background_class_id=0)

    # Expectation: [1, 2]. The 0s are background. The 1s are length 5 (>=5). The 2s are length 6 (>=5).
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"

    print("Utils validation passed.")


def validate_data_loading():
    """
    Validates data loading pipeline.
    """
    print("\nValidating Data Loading...")

    # Initialize Loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Check batch structure
    # batch: (padded_skel, padded_audio, padded_labels, lengths)
    for batch in train_loader:
        skel, audio, labels, lengths = batch

        # Validate Shapes
        # Skel: (B, T, 60)
        assert skel.dim() == 3
        assert skel.shape[2] == Config.SKELETON_INPUT_DIM

        # Audio: (B, T, 64)
        assert audio.dim() == 3
        assert audio.shape[2] == Config.AUDIO_INPUT_DIM

        # Labels: (B, T)
        assert labels.dim() == 2

        # Lengths: (B,)
        assert lengths.dim() == 1
        assert lengths.shape[0] == skel.shape[0]

        print(f"Batch shapes verified: Skel {skel.shape}, Audio {audio.shape}")
        break  # Check one batch only

    print("Data loading validation passed.")
    return train_loader


def validate_model(train_loader):
    """
    Validates model instantiation and forward pass.
    """
    print("\nValidating Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVAIIN().to(device)

    # Get a batch
    skel, audio, _, lengths = next(iter(train_loader))
    skel = skel.to(device)
    audio = audio.to(device)
    lengths = lengths.to(device)

    # Forward Pass
    outputs = model(skel, audio, lengths)

    # Output shape should be (B, T, NumClasses)
    assert outputs.dim() == 3
    assert outputs.shape[0] == skel.shape[0]
    assert outputs.shape[1] == skel.shape[1]
    assert outputs.shape[2] == Config.NUM_CLASSES

    print(f"Model forward pass successful. Output shape: {outputs.shape}")


def run_training_pipeline():
    """
    Runs the Trainer to demonstrate training loop and submission generation.
    """
    print("\nRunning Training Pipeline...")

    # Initialize Trainer
    trainer = Trainer()

    # Run training (Configured for 2 epochs)
    trainer.run()

    # Generate Submission
    trainer.generate_submission()

    # Verify Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Check content
    with open(submission_path, "r") as f:
        lines = f.readlines()
        # We used 10 test samples, so expect 10 lines
        assert len(lines) == 10, f"Expected 10 lines in submission, got {len(lines)}"

    print(f"Pipeline finished successfully. Submission saved to {submission_path}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup Environment (Data Subsetting & Config Override)
    setup_demo_environment()

    # 2. Validate Utilities
    validate_utils()

    # 3. Validate Data Loading
    loader = validate_data_loading()

    # 4. Validate Model
    validate_model(loader)

    # 5. Run Full Training & Inference Pipeline
    run_training_pipeline()

    print("\nAll demonstrations completed successfully.")
