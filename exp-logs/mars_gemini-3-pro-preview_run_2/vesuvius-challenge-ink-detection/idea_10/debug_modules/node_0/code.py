import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encoding, fbeta_score
from library.dataset import InkDataset
from library.architecture import SegFormerMiTB4
from library.losses import BCEDiceLoss
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # 1. Configuration Overrides for Speed and Demo Purposes
    print("\n[1] Configuring environment...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Use 0 for simple single-process debugging/demo
    Config.VALIDATION_THRESHOLD = -1.0  # Force save regardless of score
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.setup()  # Create cache dir

    # Set seed for reproducibility
    set_seed(42)

    # 2. Prepare Subset Metadata for Fast Training
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Select a small subset (e.g., 4 samples) to ensure the epoch finishes quickly.
    # We filter for one fragment to minimize the overhead of loading 3D volumes.
    # Assuming fragment '1' exists in train.
    subset_train = full_train_df[full_train_df["fragment_id"] == 1].head(4)
    if len(subset_train) == 0:
        # Fallback if fragment 1 isn't there, just take head
        subset_train = full_train_df.head(4)

    subset_val = full_val_df.head(2)

    # Save these subsets to working directory
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)

    print(
        f"    Saved subset train metadata ({len(subset_train)} rows) to {demo_train_path}"
    )
    print(f"    Saved subset val metadata ({len(subset_val)} rows) to {demo_val_path}")

    # Point Config to these new files
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VALIDATION_METADATA_PATH = demo_val_path

    # 3. Verify Dataset Logic
    print("\n[3] Verifying InkDataset...")
    dataset = InkDataset(subset_train, mode="train", load_cached_data=True)
    sample = dataset[0]

    # Check keys
    expected_keys = {"image", "mask", "label", "fragment_id", "coords"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Dataset sample missing keys. Found: {sample.keys()}"

    # Check Shapes
    # Image: (3, 512, 512) -> Config.IN_CHANNELS is 3
    img_shape = sample["image"].shape
    assert img_shape == (3, 512, 512), f"Unexpected image shape: {img_shape}"

    # Label: (1, 512, 512)
    lbl_shape = sample["label"].shape
    assert lbl_shape == (1, 512, 512), f"Unexpected label shape: {lbl_shape}"

    # Check Value Ranges
    assert (
        sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0
    ), "Image values out of range [0, 1]"
    unique_labels = torch.unique(sample["label"])
    assert all(
        x in [0, 1] for x in unique_labels
    ), f"Label contains non-binary values: {unique_labels}"

    print("    Dataset verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying SegFormerMiTB4 Model...")
    model = SegFormerMiTB4()
    model.eval()

    # Create a dummy batch: (Batch=2, Channels=3, H=512, W=512)
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (2, 1, 512, 512) - Logits
    assert output.shape == (
        2,
        1,
        512,
        512,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("    Model forward pass successful.")

    # 5. Verify Loss Function
    print("\n[5] Verifying BCEDiceLoss...")
    criterion = BCEDiceLoss()
    dummy_targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss = criterion(output, dummy_targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"    Loss calculation successful. Value: {loss.item():.4f}")

    # 6. Run Training Loop
    print("\n[6] Running Training Loop (Trainer)...")
    trainer = Trainer()

    # Verify internal loaders picked up the subset files
    assert len(trainer.train_dataset) == len(
        subset_train
    ), "Trainer did not load subset train data"

    # Run fit
    trainer.fit()

    # Verify model checkpoint exists
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), f"Best model not found at {best_model_path}"
    print("    Training complete and model saved.")

    # 7. Run Inference
    print("\n[7] Running Inference...")
    # Ensure test metadata exists (it should based on problem desc)
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError("Test metadata file missing.")

    run_inference()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(sub_df)} rows.")

    # Basic check of submission content
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing"

    # Check if RLE is valid (string or NaN if empty)
    # Note: If predicted is empty, it might be NaN.
    # Let's check the first row if it exists.
    if len(sub_df) > 0:
        rle_val = sub_df.iloc[0]["Predicted"]
        # It's possible to have no ink, so nan is acceptable, or a string of numbers
        if pd.notna(rle_val):
            parts = str(rle_val).split()
            assert (
                len(parts) % 2 == 0
            ), "RLE string must have even number of elements (start length pairs)"
            # Check if all are integers
            assert all(p.isdigit() for p in parts), "RLE contains non-digit characters"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
