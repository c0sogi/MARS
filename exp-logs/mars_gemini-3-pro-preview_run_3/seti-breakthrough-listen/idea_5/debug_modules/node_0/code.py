import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.utils import Config, set_seed
from library.data import TechnosignatureDataset
from library.model import SiameseSpatialFusionNet
from library.engine import ModelEngine


def main():
    # --- 1. Setup & Configuration ---
    print("Initializing Demonstration...")
    set_seed(42)

    # Define paths for the demo
    DEMO_METADATA_DIR = "./working/demo_metadata"
    DEMO_WORKING_DIR = "./working/demo_run"

    # Create directories
    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # --- 2. Prepare Mini-Datasets ---
    print("Creating mini-datasets for speed...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (e.g., 32 samples for train, 16 for val, 16 for test)
    # We use enough to form a couple of batches
    mini_train = orig_train.head(32).copy()
    mini_val = orig_val.head(16).copy()
    mini_test = orig_test.head(16).copy()

    # Save mini metadata
    mini_train.to_csv(os.path.join(DEMO_METADATA_DIR, "train.csv"), index=False)
    mini_val.to_csv(os.path.join(DEMO_METADATA_DIR, "val.csv"), index=False)
    mini_test.to_csv(os.path.join(DEMO_METADATA_DIR, "test.csv"), index=False)

    print(f"Mini-train size: {len(mini_train)}")
    print(f"Mini-val size: {len(mini_val)}")
    print(f"Mini-test size: {len(mini_test)}")

    # --- 3. Override Library Configuration ---
    # We modify the Config class attributes directly to influence the library modules
    print("Overriding Config for demo...")
    Config.METADATA_DIR = DEMO_METADATA_DIR
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.INPUT_DIR = "./input"  # Ensure this remains correct

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = (
        0  # Use 0 for simple script execution to avoid multiprocess overhead
    )
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety

    # --- 4. Verify Dataset Logic ---
    print("Verifying TechnosignatureDataset...")
    ds = TechnosignatureDataset(
        metadata_path=os.path.join(DEMO_METADATA_DIR, "train.csv"), data_type="train"
    )

    # Fetch one sample
    (img_on, img_off), target = ds[0]

    # Check types
    assert isinstance(img_on, torch.Tensor), "Image On must be a tensor"
    assert isinstance(img_off, torch.Tensor), "Image Off must be a tensor"
    assert isinstance(target, torch.Tensor), "Target must be a tensor"

    # Check shapes
    # Expected: (3, 224, 224) because Albumentations ToTensorV2 is used which is CHW,
    # and the input to transform is (H, W, 3).
    # The Config.IMAGE_SIZE is (224, 224).
    expected_shape = (3, 224, 224)
    assert (
        img_on.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {img_on.shape}"
    assert (
        img_off.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {img_off.shape}"

    print("Dataset verification passed.")

    # --- 5. Verify Model Logic ---
    print("Verifying SiameseSpatialFusionNet...")
    model = SiameseSpatialFusionNet()
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_on = torch.randn(batch_size, 3, 224, 224)
    dummy_off = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_on, dummy_off)

    # Check output shape (Batch, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"

    print("Model verification passed.")

    # --- 6. Run Full Pipeline (Engine) ---
    print("Running ModelEngine (Train -> Val -> Test)...")

    # Instantiate engine
    engine = ModelEngine()

    # Run training and inference
    # This will use the overridden Config values (1 epoch, mini datasets)
    engine.run()

    # --- 7. Verify Submission ---
    print("Verifying submission output...")
    submission_path = os.path.join(engine.submission_dir, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check length
    assert len(df_sub) == len(
        mini_test
    ), f"Submission length {len(df_sub)} does not match test set {len(mini_test)}"

    # Check columns
    assert "id" in df_sub.columns, "id column missing in submission"
    assert "target" in df_sub.columns, "target column missing in submission"

    # Check values are probabilities
    assert df_sub["target"].min() >= 0.0, "Probabilities should be >= 0"
    assert df_sub["target"].max() <= 1.0, "Probabilities should be <= 1"

    print("Submission verification passed.")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
