import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_decode, compute_levenshtein_ratio
from library.data_loader import (
    GestureDataset,
    collate_fn,
    compute_global_stats,
    get_dataloaders,
)
from library.model import KAMTRN
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration of Solution Code ===")

    # 1. Setup Demo Environment and Override Config
    print("\n[1] Setting up demo environment and configuration...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache_demo")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Create directories based on new config
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Reduce compute load
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.EARLY_STOPPING_PATIENCE = 1

    # 2. Prepare Data Subsets
    print("\n[2] Preparing data subsets for fast execution...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (take top 5 samples)
    # Ensure we pick samples that actually exist to avoid errors
    train_subset = orig_train.head(5).copy()
    val_subset = orig_val.head(5).copy()
    test_subset = orig_test.head(5).copy()

    # Save subsets
    train_subset_path = os.path.join(demo_dir, "train_subset.csv")
    val_subset_path = os.path.join(demo_dir, "val_subset.csv")
    test_subset_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # Point Config to subsets
    Config.TRAIN_CSV = train_subset_path
    Config.VAL_CSV = val_subset_path
    Config.TEST_CSV = test_subset_path

    print(f"    Train subset: {len(train_subset)} samples")
    print(f"    Val subset:   {len(val_subset)} samples")
    print(f"    Test subset:  {len(test_subset)} samples")

    # 3. Verify Utils
    print("\n[3] Verifying Utility Functions...")

    # Test rle_decode
    # Create a dummy probability sequence: 10 frames of class 1, 10 frames of class 2
    # Config.MIN_SEGMENT_LENGTH is 5 by default
    T = 20
    probs = np.zeros((T, Config.NUM_CLASSES))
    probs[0:10, 1] = 1.0  # Class 1
    probs[10:20, 2] = 1.0  # Class 2

    decoded = rle_decode(probs, min_length=5, background_class=0)
    print(f"    RLE Decode Result: {decoded}")
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"

    # Test compute_levenshtein_ratio
    # Perfect match
    score_perfect = compute_levenshtein_ratio([[1, 2]], [[1, 2]])
    assert score_perfect == 0.0, "Perfect match should have 0 error"

    # Complete mismatch (substitution)
    # Distance between [1] and [2] is 1. Length of truth is 1. Ratio = 1.0
    score_mismatch = compute_levenshtein_ratio([[1]], [[2]])
    assert abs(score_mismatch - 1.0) < 1e-6, "Mismatch should have error 1.0"

    print("    Utils verification passed.")

    # 4. Verify Data Loader
    print("\n[4] Verifying Data Loader...")

    # Compute stats on subset
    stats = compute_global_stats(train_subset)
    assert "pose_mean" in stats
    assert stats["pose_mean"].shape[0] == Config.POSE_INPUT_DIM

    # Instantiate Dataset
    ds = GestureDataset(train_subset, mode="train", stats=stats, load_cached_data=True)

    # Check __getitem__
    pose, velocity, audio, labels, boundaries = ds[0]
    print(
        f"    Sample 0 Shapes -> Pose: {pose.shape}, Vel: {velocity.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    assert pose.shape[1] == Config.POSE_INPUT_DIM
    assert velocity.shape[1] == Config.VELOCITY_INPUT_DIM
    assert audio.shape[1] == Config.AUDIO_INPUT_DIM
    assert pose.shape[0] == labels.shape[0]  # Time dimension match

    # Check DataLoader and Collate
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dl))
    b_pose, b_vel, b_audio, b_labels, b_bound, b_lens = batch

    print(f"    Batch Shapes -> Pose: {b_pose.shape} (Batch, Time, Feat)")
    assert b_pose.shape[0] == 2
    assert b_pose.shape[2] == Config.POSE_INPUT_DIM
    assert b_lens.shape[0] == 2

    print("    Data Loader verification passed.")

    # 5. Verify Model
    print("\n[5] Verifying Model Architecture...")

    device = Config.get_device()
    model = KAMTRN().to(device)

    # Move batch to device
    b_pose = b_pose.to(device)
    b_vel = b_vel.to(device)
    b_audio = b_audio.to(device)

    # Forward pass
    class_logits, boundary_logits = model(b_pose, b_vel, b_audio)

    print(
        f"    Output Shapes -> Class Logits: {class_logits.shape}, Boundary Logits: {boundary_logits.shape}"
    )

    assert class_logits.shape[0] == 2
    assert class_logits.shape[2] == Config.NUM_CLASSES
    assert boundary_logits.shape[0] == 2
    assert boundary_logits.shape[2] == 1
    assert class_logits.shape[1] == b_pose.shape[1]  # Time dimension preserved

    print("    Model verification passed.")

    # 6. Verify Trainer Loop
    print("\n[6] Verifying Training Loop...")

    # Instantiate Trainer
    trainer = Trainer()

    # Get loaders (using the subsets configured in Config)
    train_loader, val_loader, test_loader = get_dataloaders()

    # Run one training epoch step manually
    print("    Running single training epoch...")
    train_loss = trainer.train_epoch(train_loader)
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation step manually
    print("    Running validation...")
    val_score, val_loss = trainer.validate(val_loader)
    print(f"    Val Loss: {val_loss:.4f}, Levenshtein Error: {val_score:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert val_score >= 0, "Levenshtein error must be non-negative"

    # Save a dummy checkpoint to test prediction loading
    torch.save(
        trainer.model.state_dict(),
        os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    )

    print("    Trainer loop verification passed.")

    # 7. Verify Prediction
    print("\n[7] Verifying Prediction...")

    # Run prediction method
    trainer.predict()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    with open(Config.SUBMISSION_PATH, "r") as f:
        submission_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"    Submission rows: {len(submission_lines)}")
    if len(submission_lines) > 0:
        print(f"    First row: {submission_lines[0]}")

    # We expect 5 rows because test_subset has 5 samples
    assert (
        len(submission_lines) == 5
    ), f"Expected 5 predictions, got {len(submission_lines)}"

    print("    Prediction verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
