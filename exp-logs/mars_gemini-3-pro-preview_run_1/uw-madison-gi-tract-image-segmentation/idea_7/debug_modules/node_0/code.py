import os
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library
from library import config
from library import utils
from library import dataset
from library import model
from library import train
from library import evaluate


def run_demo():
    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    print("Seed set to:", config.SEED)

    # ==========================================
    # 1. Verify Utility Functions
    # ==========================================
    print("\n=== 1. Verifying Utility Functions ===")

    # Test RLE Encoding/Decoding
    # Create a simple 10x10 mask with a 2x2 square in the middle
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[4:6, 4:6] = 1

    rle_str = utils.rle_encode(dummy_mask)
    decoded_mask = utils.rle_decode(rle_str, (10, 10))

    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Decode does not match original mask"
    print("RLE Encode/Decode: PASSED")

    # Test Percentile Normalization
    # Create random image with outliers
    dummy_img = np.random.randint(0, 255, (100, 100)).astype(np.float32)
    dummy_img[0, 0] = 10000.0  # Outlier
    norm_img = utils.percentile_normalize(dummy_img, lower=1.0, upper=99.0)

    assert (
        norm_img.min() >= 0.0 and norm_img.max() <= 1.0
    ), "Normalized image out of bounds [0, 1]"
    assert norm_img.dtype == np.float32, "Normalized image is not float32"
    print("Percentile Normalization: PASSED")

    # Test Dice Coefficient
    mask_a = np.ones((10, 10))
    mask_b = np.zeros((10, 10))
    dice_score = utils.compute_dice(mask_a, mask_a)
    assert (
        dice_score == 1.0
    ), f"Dice score for identical masks should be 1.0, got {dice_score}"

    dice_score_empty = utils.compute_dice(mask_b, mask_b)
    assert (
        dice_score_empty == 0.0
    ), f"Dice score for empty masks should be 0.0, got {dice_score_empty}"
    print("Dice Metric Calculation: PASSED")

    # ==========================================
    # 2. Verify Dataset Loading
    # ==========================================
    print("\n=== 2. Verifying Dataset Loading ===")

    # Load metadata
    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {config.TRAIN_CSV}")

    df_train = pd.read_csv(config.TRAIN_CSV, keep_default_na=False)

    # Use a tiny subset for quick verification
    # Filter for a specific case to ensure we have enough slices for a sequence
    case_id = df_train["case"].unique()[0]
    df_subset = df_train[df_train["case"] == case_id].head(20)

    print(
        f"Creating dataset from subset of case {case_id} ({len(df_subset)} slices)..."
    )

    # Initialize Dataset
    ds = dataset.SliceSequenceDataset(df_subset, mode="train", load_cached_data=False)

    assert len(ds) > 0, "Dataset is empty"

    # Fetch one sample
    img_tensor, mask_tensor = ds[len(ds) // 2]

    # Check Shapes
    # Image: (Seq, 3, H, W) -> (5, 3, 256, 256) based on config
    expected_img_shape = (config.SEQ_LEN, 3, config.IMG_SIZE[0], config.IMG_SIZE[1])
    # Mask: (C, H, W) -> (3, 256, 256)
    expected_mask_shape = (config.NUM_CLASSES, config.IMG_SIZE[0], config.IMG_SIZE[1])

    assert (
        img_tensor.shape == expected_img_shape
    ), f"Image tensor shape mismatch. Expected {expected_img_shape}, got {img_tensor.shape}"
    assert (
        mask_tensor.shape == expected_mask_shape
    ), f"Mask tensor shape mismatch. Expected {expected_mask_shape}, got {mask_tensor.shape}"
    assert img_tensor.dtype == torch.float32, "Image tensor is not float32"

    print(
        f"Dataset Item Shapes: Image {tuple(img_tensor.shape)}, Mask {tuple(mask_tensor.shape)}"
    )
    print("Dataset Verification: PASSED")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n=== 3. Verifying Model Architecture ===")

    # Initialize Model
    net = model.RecurrentUNet(
        num_classes=config.NUM_CLASSES, seq_len=config.SEQ_LEN, pretrained=False
    )
    net.to(config.DEVICE)
    net.eval()

    # Create dummy batch: (Batch, Seq, C, H, W)
    batch_size = 2
    dummy_input = torch.randn(
        batch_size, config.SEQ_LEN, 3, config.IMG_SIZE[0], config.IMG_SIZE[1]
    ).to(config.DEVICE)

    with torch.no_grad():
        output = net(dummy_input)

    # Expected Output: (Batch, Num_Classes, H, W)
    expected_out_shape = (
        batch_size,
        config.NUM_CLASSES,
        config.IMG_SIZE[0],
        config.IMG_SIZE[1],
    )

    assert (
        output.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"
    print(f"Model Forward Pass Output Shape: {tuple(output.shape)}")
    print("Model Verification: PASSED")

    # ==========================================
    # 4. Run Training Demo
    # ==========================================
    print("\n=== 4. Running Training Demo (Debug Mode) ===")

    # Clean up previous checkpoints if any (for demo purposes)
    if os.path.exists(config.CHECKPOINT_DIR):
        # We don't delete the dir to avoid permission issues, just check file
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(ckpt_path):
            print("Removing existing checkpoint for fresh demo...")
            os.remove(ckpt_path)

    # Run training for 1 epoch with debug=True (uses small data subset)
    # This function handles DataLoader creation, Model init, and Training Loop
    train.train_model(debug=True, epochs=1)

    # Verify Checkpoint Creation
    expected_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        expected_ckpt
    ), f"Checkpoint file was not created at {expected_ckpt}"
    print(f"Checkpoint successfully created at: {expected_ckpt}")
    print("Training Demo: PASSED")

    # ==========================================
    # 5. Run Evaluation & Submission Demo
    # ==========================================
    print("\n=== 5. Running Evaluation & Submission Demo ===")

    # This runs inference on Val (calculates metrics) and Test (generates submission)
    # debug=True restricts the number of samples processed
    evaluate.run_evaluation(debug=True)

    # Verify Submission File
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    # Check submission content format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    required_cols = ["id", "class", "predicted"]

    assert all(
        col in sub_df.columns for col in required_cols
    ), f"Submission missing columns. Found: {sub_df.columns}"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"Submission generated with {len(sub_df)} rows.")
    print(f"First row: {sub_df.iloc[0].to_dict()}")
    print("Evaluation & Submission Demo: PASSED")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
