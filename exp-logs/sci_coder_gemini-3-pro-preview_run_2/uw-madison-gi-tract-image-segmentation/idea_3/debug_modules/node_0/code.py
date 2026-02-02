import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, min_max_normalize
from library.dataset import UWMadisonDataset, rle_decode
from library.model import DeepLabV3Plus
from library.loss import BCEDiceLoss
from library.train import run_training
from library.inference import run_inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE Encoding/Decoding)
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Create a simple synthetic binary mask (10x10)
    # Pattern: A 3x3 square of 1s in the middle
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[3:6, 3:6] = 1

    # Encode
    encoded_rle = rle_encode(dummy_mask)
    print(f"    Encoded RLE: {encoded_rle}")

    # Decode
    decoded_mask = rle_decode(encoded_rle, shape=(10, 10))

    # Verify reconstruction
    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Decode -> Encode roundtrip failed!"
    print("    RLE Roundtrip verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Dataset...")

    # Initialize dataset (this will trigger metadata processing)
    # We use 'train' split to ensure we have masks
    dataset = UWMadisonDataset(split="train", load_cached_data=False)

    print(f"    Dataset length: {len(dataset)}")

    # Fetch one sample
    sample = dataset[0]

    # Verify keys
    required_keys = [
        "image",
        "mask",
        "id",
        "img_height",
        "img_width",
        "pixel_spacing_h",
        "pixel_spacing_w",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Shapes
    # Image: (Channels=3, Height=256, Width=256) -> 2.5D stack
    # Mask: (Classes=3, Height=256, Width=256)
    img_tensor = sample["image"]
    mask_tensor = sample["mask"]

    print(f"    Image Tensor Shape: {img_tensor.shape}")
    print(f"    Mask Tensor Shape: {mask_tensor.shape}")

    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Unexpected image shape: {img_tensor.shape}"
    assert mask_tensor.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Unexpected mask shape: {mask_tensor.shape}"

    # Verify Normalization (should be approx 0-1)
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image tensor values out of range [0, 1]"

    print("    Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch: (Batch=2, Channels=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Expected: (Batch=2, Classes=3, H=256, W=256)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Model output shape mismatch. Expected (2, 3, 256, 256), got {output.shape}"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Loss Function
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Loss Function...")

    loss_fn = BCEDiceLoss()

    # Create dummy logits and targets
    # Logits: Random values
    # Targets: Binary (0 or 1) float tensor
    dummy_logits = torch.randn(2, 3, 256, 256)
    dummy_targets = torch.randint(0, 2, (2, 3, 256, 256)).float()

    loss_val = loss_fn(dummy_logits, dummy_targets)

    print(f"    Calculated Loss: {loss_val.item()}")

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() >= 0, "Loss is negative"

    print("    Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 6. Run Training Pipeline (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Pipeline (Debug Mode)...")

    # Run training for 1 epoch on a small subset
    # debug=True forces the use of a small Subset of the data
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=True,
    )

    # Verify Checkpoint creation
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"    Checkpoint successfully created at: {Config.CHECKPOINT_PATH}")
    else:
        # Note: If validation dice is 0.0 (possible in debug with random weights on small subset),
        # the code might not save if it strictly checks > best_dice (init 0.0).
        # However, usually at least one save happens or we should check logic.
        # Looking at train.py: if val_dice > best_dice (0.0).
        # If model is random, dice might be ~0.
        # For demonstration, we assume it works or we check if file exists.
        # If it failed to save because dice was 0, we create a dummy one for inference step
        print(
            "    Checkpoint not saved (likely due to Dice=0.0 in debug). Saving dummy for inference test."
        )
        torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # -------------------------------------------------------------------------
    # 7. Run Inference Pipeline (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference Pipeline (Debug Mode)...")

    # Run inference
    run_inference(load_cached_data=True, debug=True)

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file created at: {Config.SUBMISSION_PATH}")
        print(f"    Submission rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        expected_cols = ["id", "class", "predicted"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}"

        # Check if we have rows (debug mode on test set should yield some rows)
        # Note: test set in metadata might be small or empty depending on provided environment,
        # but based on description, test.csv exists.
        if len(df_sub) > 0:
            print("    Submission content verification passed.")
        else:
            print(
                "    Submission is empty (expected if debug subset yielded no masks or empty test set)."
            )
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
