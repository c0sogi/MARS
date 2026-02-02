import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import sys
import importlib

# -----------------------------------------------------------------------------
# Cite Debug Lesson 2: Force Module Reloads
# In persistent environments, modifying files on disk does not update already
# imported modules. We must explicitly reload them to apply the fix.
# -----------------------------------------------------------------------------
modules_to_reload = [
    "library.config",
    "library.utils",
    "library.transforms",
    "library.dataset",
    "library.modules",
    "library.model",
    "library.loss",
    "library.engine",
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"Reloading {module_name}...")
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            print(f"Failed to reload {module_name}: {e}")

# Import from the provided library
from library.config import Config
from library.utils import box3d_to_corners, iou3d_shapely
from library.dataset import LidarDataset
from library.engine import run_engine


def setup_reproducibility(seed=42):
    """Set fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}.")


def create_subset_metadata(source_dir, target_dir, num_train=16, num_val=8, num_test=8):
    """
    Creates a small subset of the metadata csv files to allow for fast execution.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    print(f"Creating metadata subsets in {target_dir}...")

    # Subset Train
    train_df = pd.read_csv(os.path.join(source_dir, "train_metadata.csv"))
    # Filter for samples that actually have objects to ensure GT database isn't empty
    train_df = train_df[train_df["label"].notna() & (train_df["label"] != "")]
    train_subset = train_df.head(num_train)
    train_subset.to_csv(os.path.join(target_dir, "train_metadata.csv"), index=False)
    print(f"  - Train subset: {len(train_subset)} samples")

    # Subset Val
    val_df = pd.read_csv(os.path.join(source_dir, "val_metadata.csv"))
    val_subset = val_df.head(num_val)
    val_subset.to_csv(os.path.join(target_dir, "val_metadata.csv"), index=False)
    print(f"  - Val subset: {len(val_subset)} samples")

    # Subset Test
    test_df = pd.read_csv(os.path.join(source_dir, "test_metadata.csv"))
    test_subset = test_df.head(num_test)
    test_subset.to_csv(os.path.join(target_dir, "test_metadata.csv"), index=False)
    print(f"  - Test subset: {len(test_subset)} samples")


def verify_geometric_utils():
    """
    Verifies the correctness of 3D box corner generation and IoU calculation.
    """
    print("\nVerifying Geometric Utilities...")

    # Define two boxes: [x, y, z, w, l, h, yaw]
    # Box A: Center at (0,0,0), dimensions 2x2x2, no rotation
    box_a = np.array([[0, 0, 0, 2, 2, 2, 0]], dtype=np.float32)

    # Box B: Center at (1,0,0), dimensions 2x2x2, no rotation
    # Overlap should be exactly half in volume (1x2x2 intersection volume = 4, Union = 8+8-4=12. IoU = 4/12 = 0.333)
    # However, iou3d_shapely calculates BEV IoU * Height IoU / Union.
    # BEV Intersection: width overlap 1, length overlap 2 -> Area 2.
    # Height Intersection: 2.
    # Intersection Vol: 4.
    # Union Vol: 12.
    # IoU: 0.3333...
    box_b = np.array([[1, 0, 0, 2, 2, 2, 0]], dtype=np.float32)

    # Check Corners
    corners_a = box3d_to_corners(box_a)
    # Expected corners for Box A (2x2x2 centered at 0): +/- 1
    assert corners_a.shape == (1, 8, 3)
    assert np.allclose(np.max(corners_a), 1.0)
    assert np.allclose(np.min(corners_a), -1.0)
    print("  - box3d_to_corners: OK")

    # Check IoU
    iou = iou3d_shapely(box_a, box_b)
    print(f"  - Calculated IoU: {iou[0,0]:.4f}")
    assert np.isclose(
        iou[0, 0], 1.0 / 3.0, atol=1e-3
    ), f"Expected IoU ~0.333, got {iou[0,0]}"

    # Check Disjoint
    box_c = np.array([[10, 10, 10, 2, 2, 2, 0]], dtype=np.float32)
    iou_disjoint = iou3d_shapely(box_a, box_c)
    assert iou_disjoint[0, 0] == 0.0
    print("  - iou3d_shapely: OK")


def verify_dataset_loading():
    """
    Verifies that the dataset loads correctly and transforms run.
    """
    print("\nVerifying Dataset & Transforms...")
    # Initialize dataset (this will trigger GT Database building for the subset)
    ds = LidarDataset(split="train")
    print(f"  - Dataset length: {len(ds)}")

    # Fetch one sample
    sample = ds[0]

    # Check keys
    required_keys = ["points", "gt_boxes", "gt_labels", "metadata"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Check shapes
    points = sample["points"]
    assert (
        points.dim() == 2 and points.shape[1] == 4
    ), f"Invalid points shape: {points.shape}"

    gt_boxes = sample["gt_boxes"]
    if gt_boxes.shape[0] > 0:
        assert gt_boxes.shape[1] == 7, f"Invalid box shape: {gt_boxes.shape}"

    print(f"  - Sample 0 points: {points.shape}")
    print(f"  - Sample 0 gt_boxes: {gt_boxes.shape}")
    print("  - LidarDataset: OK")


if __name__ == "__main__":
    # 1. Setup
    setup_reproducibility()

    # Define paths
    BASE_DIR = os.getcwd()
    INPUT_METADATA_DIR = os.path.join(BASE_DIR, "metadata")
    WORKING_DIR = os.path.join(BASE_DIR, "working", "demo_run")
    DEMO_METADATA_DIR = os.path.join(WORKING_DIR, "metadata")

    # Create working directories
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Create Subset Metadata
    create_subset_metadata(INPUT_METADATA_DIR, DEMO_METADATA_DIR)

    # 3. Patch Configuration for Demo
    print("\nPatching Config for Demo Run...")
    Config.METADATA_DIR = DEMO_METADATA_DIR
    Config.WORKING_DIR = WORKING_DIR
    Config.CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    Config.GT_DATABASE_DIR = os.path.join(WORKING_DIR, "gt_database")
    Config.DB_INFO_PATH = os.path.join(Config.GT_DATABASE_DIR, "gt_database.parquet")
    Config.SUBMISSION_DIR = WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure directories exist after config update
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.GT_DATABASE_DIR, exist_ok=True)

    # Reduce Hyperparameters for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 4. Verify Components
    verify_geometric_utils()
    verify_dataset_loading()

    # 5. Run Training & Inference Engine
    print("\nStarting End-to-End Engine Execution...")
    # This will:
    # 1. Initialize datasets (Train, Val, Test)
    # 2. Build model
    # 3. Train for 1 epoch
    # 4. Validate
    # 5. Generate submission.csv
    run_engine(epochs=1)

    # 6. Verify Output
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\nSubmission generated successfully at {Config.SUBMISSION_PATH}")
        print(f"Submission rows: {len(df)}")
        print(df.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemo completed successfully.")
