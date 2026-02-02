import os
import pandas as pd
import numpy as np
import torch
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import read_bson_images
from library.dataset import CdiscountDataset, collate_flatten
from library.model import get_model
import library.engine as engine


def setup_demo_environment():
    """
    Creates a small subset of the metadata to allow for rapid demonstration
    and testing of the pipeline without processing the entire 58GB dataset.
    """
    print("Setting up demo environment...")

    # Create working directories
    demo_dir = "./working/demo_data"
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Create a small subset of Train/Val metadata
    # We read the first 200 rows from the generated metadata
    full_train_meta_path = Config.TRAIN_META
    if not os.path.exists(full_train_meta_path):
        raise FileNotFoundError(
            f"Original metadata not found at {full_train_meta_path}"
        )

    df_full = pd.read_csv(full_train_meta_path, nrows=200)

    # Split into demo train (160) and demo val (40)
    df_demo_train = df_full.iloc[:160].copy()
    df_demo_val = df_full.iloc[160:].copy()

    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")

    df_demo_train.to_csv(demo_train_path, index=False)
    df_demo_val.to_csv(demo_val_path, index=False)

    # 2. Create a small subset of Test metadata
    full_test_meta_path = Config.TEST_META
    if not os.path.exists(full_test_meta_path):
        raise FileNotFoundError(
            f"Original test metadata not found at {full_test_meta_path}"
        )

    df_test_full = pd.read_csv(full_test_meta_path, nrows=50)
    demo_test_path = os.path.join(demo_dir, "test_subset.csv")
    df_test_full.to_csv(demo_test_path, index=False)

    print(
        f"Created subset metadata: \n  Train: {len(df_demo_train)}\n  Val: {len(df_demo_val)}\n  Test: {len(df_test_full)}"
    )

    return demo_train_path, demo_val_path, demo_test_path


def patch_config(train_path, val_path, test_path):
    """
    Overwrites Config attributes to use the small datasets and reduce runtime parameters.
    """
    print("Patching Config for demo run...")

    # Point to subset metadata
    Config.TRAIN_META = train_path
    Config.VAL_META = val_path
    Config.TEST_META = test_path

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch for the demo
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Output paths
    Config.WORKING_DIR = "./working/demo_output"
    Config.MODEL_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    os.makedirs(Config.MODEL_CHECKPOINT_DIR, exist_ok=True)


def verify_bson_reading():
    """
    Verifies that the utility function can correctly read images from the BSON file
    using the metadata offsets.
    """
    print("\n[Verification] Testing BSON Image Reading...")

    # Pick a row from our demo train set
    df = pd.read_csv(Config.TRAIN_META)
    row = df.iloc[0]

    bson_path = os.path.join(Config.INPUT_DIR, row["file_path"])
    offset = row["bson_offset"]
    length = row["bson_length"]

    # Use the library function
    images = read_bson_images(bson_path, offset, length)

    # Assertions
    if not isinstance(images, list):
        raise AssertionError("read_bson_images should return a list.")
    if len(images) == 0:
        raise AssertionError("No images returned for the sample record.")

    img = images[0]
    if not isinstance(img, np.ndarray):
        raise AssertionError("Image should be a numpy array.")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise AssertionError(f"Image should be RGB (H, W, 3). Got shape {img.shape}")

    print(
        f"Successfully read {len(images)} image(s) from product {row['product_id']}. Shape: {img.shape}"
    )


def verify_dataset_and_collate():
    """
    Verifies the Dataset class and the custom collate function.
    """
    print("\n[Verification] Testing Dataset and Collate Function...")

    # Initialize Dataset
    ds = CdiscountDataset(Config.TRAIN_META, mode="train")

    # 1. Test __getitem__
    images_tensor, target, product_id = ds[0]

    # Expect (N, 3, 224, 224)
    if len(images_tensor.shape) != 4:
        raise AssertionError(
            f"Expected 4D tensor (N, C, H, W), got {images_tensor.shape}"
        )
    if images_tensor.shape[1] != 3 or images_tensor.shape[2] != 224:
        raise AssertionError(f"Expected 3x224x224 images, got {images_tensor.shape}")
    if not isinstance(target, int):
        raise AssertionError("Target should be an integer.")

    print(
        f"Dataset item 0: Tensor {images_tensor.shape}, Target {target}, PID {product_id}"
    )

    # 2. Test DataLoader with collate_flatten
    # We use a batch size of 4
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_flatten)
    batch_images, batch_targets, batch_pids = next(iter(loader))

    # Check flattening
    # Since products have variable num images, batch_images.shape[0] >= batch_size
    if batch_images.shape[0] < 4:
        raise AssertionError("Batch size mismatch after flattening.")
    if batch_images.shape[1:] != (3, 224, 224):
        raise AssertionError("Image dimensions corrupted in batch.")
    if batch_targets.shape[0] != batch_images.shape[0]:
        raise AssertionError("Targets dimension must match flattened images dimension.")

    print(f"Batch collation successful. Flattened batch size: {batch_images.shape[0]}")


def verify_model():
    """
    Verifies model instantiation and forward pass.
    """
    print("\n[Verification] Testing Model...")

    model = get_model(pretrained=False)  # False for speed
    model.eval()

    # Create dummy input: 2 images, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (2, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
        )

    print(f"Model forward pass successful. Output shape: {output.shape}")


def run_pipeline():
    """
    Runs the full training and inference pipeline using the engine module.
    """
    print("\n[Pipeline] Starting Engine Training Loop...")

    # This calls the run_training function in engine.py
    # Because we patched Config, it will use our subset data and parameters.
    engine.run_training()

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "model_best.pth")
    if not os.path.exists(checkpoint_path):
        raise AssertionError("Training did not produce a 'model_best.pth' checkpoint.")
    print(f"Checkpoint verified at {checkpoint_path}")

    print("\n[Pipeline] Starting Engine Inference Loop...")
    engine.run_inference()

    # Check if submission was created
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Inference did not produce a submission file.")

    # Validate submission format
    sub = pd.read_csv(Config.SUBMISSION_PATH)
    if list(sub.columns) != ["_id", "category_id"]:
        raise AssertionError(f"Submission columns incorrect. Got {sub.columns}")
    if len(sub) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"Submission verified at {Config.SUBMISSION_PATH}. Rows: {len(sub)}")
    print("Top 5 predictions:")
    print(sub.head())


if __name__ == "__main__":
    # Ensure reproducibility
    engine.set_seed(42)

    # 1. Setup Data
    train_csv, val_csv, test_csv = setup_demo_environment()

    # 2. Configure System
    patch_config(train_csv, val_csv, test_csv)

    # 3. Verify Components
    verify_bson_reading()
    verify_dataset_and_collate()
    verify_model()

    # 4. Run Integration
    run_pipeline()

    print("\nAll demonstrations and verifications completed successfully.")
