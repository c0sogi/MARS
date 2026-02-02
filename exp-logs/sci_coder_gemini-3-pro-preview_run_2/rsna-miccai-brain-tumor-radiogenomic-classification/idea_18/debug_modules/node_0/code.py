import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, normalize_minmax, resize_volume
from library.data import get_dataloaders
from library.model import SiameseEfficientNet
from library.train import run_training
from library.predict import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Environment Setup
    # --------------------------------------------------------------------------
    print("\n[Step 1] Setting up environment and overriding configuration...")

    # Define a temporary directory for this demo run
    demo_dir = "./working/demo_run"
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Cache paths (must be done explicitly as they are class attributes)
    Config.CACHE_TRAIN_DATA = os.path.join(demo_dir, "data_cache_train.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "labels_cache_train.npy")
    Config.CACHE_VAL_DATA = os.path.join(demo_dir, "data_cache_val.npy")
    Config.CACHE_VAL_LABELS = os.path.join(demo_dir, "labels_cache_val.npy")
    Config.CACHE_TEST_DATA = os.path.join(demo_dir, "data_cache_test.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "labels_cache_test.npy")

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = 224

    # --------------------------------------------------------------------------
    # 2. Create Subset Metadata (Speed Optimization)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Creating metadata subsets for rapid execution...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (enough for 2 batches of train, 1 of val/test)
    sub_train = orig_train.head(8).copy()
    sub_val = orig_val.head(4).copy()
    sub_test = orig_test.head(4).copy()

    # Save subsets
    sub_train_path = os.path.join(demo_meta_dir, "train.csv")
    sub_val_path = os.path.join(demo_meta_dir, "val.csv")
    sub_test_path = os.path.join(demo_meta_dir, "test.csv")

    sub_train.to_csv(sub_train_path, index=False)
    sub_val.to_csv(sub_val_path, index=False)
    sub_test.to_csv(sub_test_path, index=False)

    # Point Config to these new metadata files
    Config.TRAIN_METADATA = sub_train_path
    Config.VAL_METADATA = sub_val_path
    Config.TEST_METADATA = sub_test_path

    print(
        f"Subset sizes -> Train: {len(sub_train)}, Val: {len(sub_val)}, Test: {len(sub_test)}"
    )

    # --------------------------------------------------------------------------
    # 3. Verify Utils
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying utility functions...")
    seed_everything(Config.SEED)

    # Test normalization
    dummy_img = np.array([[0, 100], [50, 200]], dtype=np.float32)
    norm_img = normalize_minmax(dummy_img)
    assert norm_img.min() == 0.0 and norm_img.max() == 1.0, "Normalization failed"
    assert norm_img.shape == (2, 2), "Shape mismatch in normalization"

    # Test resizing
    resized = resize_volume(dummy_img, (10, 10))
    assert resized.shape == (10, 10), "Resize failed"
    print("Utils verification passed.")

    # --------------------------------------------------------------------------
    # 4. Verify Data Pipeline
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Data Pipeline (Loading & Caching)...")

    # Force reload to generate cache from our new subsets
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check cache file creation
    assert os.path.exists(Config.CACHE_TRAIN_DATA), "Train data cache not created"
    assert os.path.exists(Config.CACHE_TRAIN_LABELS), "Train labels cache not created"

    # Fetch one batch
    view_bulk, view_core, labels = next(iter(train_loader))

    # Verify shapes
    # Expected: (Batch, 12, 224, 224)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )

    print(
        f"Batch Shapes -> Bulk: {view_bulk.shape}, Core: {view_core.shape}, Labels: {labels.shape}"
    )

    assert (
        view_bulk.shape == expected_shape
    ), f"Bulk view shape mismatch. Got {view_bulk.shape}"
    assert (
        view_core.shape == expected_shape
    ), f"Core view shape mismatch. Got {view_core.shape}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("Data pipeline verification passed.")

    # --------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[Step 5] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check to avoid overhead
    model = SiameseEfficientNet().to(device)
    model.eval()

    with torch.no_grad():
        # Pass the batch retrieved from step 4
        logits = model(view_bulk.to(device), view_core.to(device))

    print(f"Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("Model verification passed.")

    # --------------------------------------------------------------------------
    # 6. Verify Training Loop
    # --------------------------------------------------------------------------
    print("\n[Step 6] Verifying Training Loop (1 Epoch)...")

    # This will use the cached data we just generated
    run_training(load_cached_data=True)

    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint was not saved after training"
    print("Training loop verification passed.")

    # --------------------------------------------------------------------------
    # 7. Verify Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[Step 7] Verifying Inference and Submission Generation...")

    generate_submission(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Incorrect submission columns"
    assert len(df_sub) == len(
        sub_test
    ), f"Submission length mismatch. Expected {len(sub_test)}, got {len(df_sub)}"
    assert df_sub["MGMT_value"].dtype == float, "MGMT_value should be float"

    print("Inference verification passed.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
