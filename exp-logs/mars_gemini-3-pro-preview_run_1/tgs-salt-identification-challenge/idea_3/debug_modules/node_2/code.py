import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    calculate_iou,
    calculate_map_score,
)
from library.dataset import SaltDataset
from library.model import HyperColumnUNet
from library.losses import BCEDiceLoss, LovaszLoss
from library.train import run_fold


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    print("--- Setting up environment ---")
    set_seed(42)
    warnings.filterwarnings("ignore")

    WORK_DIR = "./working/demo_execution"
    os.makedirs(WORK_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utils (RLE and Metrics)
    # -------------------------------------------------------------------------
    print("--- Verifying Utilities ---")

    # Create a dummy mask: 101x101 with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    # Test RLE Encoding
    rle_str = rle_encode(dummy_mask)
    assert isinstance(rle_str, str), "RLE encode should return a string"
    assert len(rle_str) > 0, "RLE string should not be empty for non-empty mask"

    # Test RLE Decoding
    decoded_mask = rle_decode(rle_str, shape=(101, 101))
    assert np.array_equal(dummy_mask, decoded_mask), "Decoded mask must match original"

    # Test IoU
    iou_perfect = calculate_iou(dummy_mask, dummy_mask)
    assert np.isclose(iou_perfect, 1.0), "IoU of identical masks should be 1.0"

    empty_mask = np.zeros((101, 101), dtype=np.uint8)
    iou_zero = calculate_iou(dummy_mask, empty_mask)
    assert np.isclose(iou_zero, 0.0), "IoU of disjoint masks should be 0.0"

    # Test mAP
    # Perfect match case
    map_score = calculate_map_score([dummy_mask], [dummy_mask])
    assert np.isclose(map_score, 1.0), "mAP for perfect match should be 1.0"

    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Preparation (Mini Subsets)
    # -------------------------------------------------------------------------
    print("--- Preparing Mini Datasets ---")

    # Load full metadata
    full_train_df = pd.read_csv("./metadata/train.csv")
    full_val_df = pd.read_csv("./metadata/val.csv")

    # Create subsets (16 train, 8 val) to ensure speed
    mini_train_df = full_train_df.head(16).copy()
    mini_val_df = full_val_df.head(8).copy()

    mini_train_path = os.path.join(WORK_DIR, "mini_train.csv")
    mini_val_path = os.path.join(WORK_DIR, "mini_val.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)

    print(f"Created mini datasets at {mini_train_path} and {mini_val_path}")

    # -------------------------------------------------------------------------
    # 4. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("--- Verifying Dataset Class ---")

    # Initialize dataset with mini metadata
    # Note: cache_dir is set to WORK_DIR to avoid polluting other dirs
    dataset = SaltDataset(
        metadata_csv=mini_train_path,
        mode="train",
        cache_dir=WORK_DIR,
        load_cached=False,  # Force reload to test loading logic
    )

    assert len(dataset) == 16, f"Dataset length should be 16, got {len(dataset)}"

    # Fetch one sample
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]

    # Check shapes
    # Expected: Image (2, 128, 128) -> 1 grayscale + 1 depth, padded from 101
    # Expected: Mask (1, 128, 128)
    assert image.shape == (2, 128, 128), f"Unexpected image shape: {image.shape}"
    assert mask.shape == (1, 128, 128), f"Unexpected mask shape: {mask.shape}"

    # Check value ranges
    assert (
        image.min() >= 0.0 and image.max() <= 1.0
    ), "Image values should be normalized 0-1"
    assert torch.all((mask == 0) | (mask == 1)), "Mask should be binary (0 or 1)"

    # Verify depth channel (channel 1) is constant per image (it's a plane)
    depth_plane = image[1, :, :]
    assert torch.std(depth_plane) < 1e-6, "Depth channel should be constant spatially"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Model and Losses
    # -------------------------------------------------------------------------
    print("--- Verifying Model and Losses ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate Model
    model = HyperColumnUNet(
        input_channels=2, num_classes=1, base_filters=16
    )  # Reduced filters for speed
    model = model.to(device)

    # Create a batch
    batch_imgs = image.unsqueeze(0).to(device)  # (1, 2, 128, 128)
    batch_masks = mask.unsqueeze(0).to(device)  # (1, 1, 128, 128)

    # Forward Pass
    output = model(batch_imgs)
    assert output.shape == (1, 1, 128, 128), f"Output shape mismatch: {output.shape}"

    # Loss Calculation
    bce_dice = BCEDiceLoss()
    lovasz = LovaszLoss()

    loss1 = bce_dice(output, batch_masks)
    loss2 = lovasz(output, batch_masks)

    assert not torch.isnan(loss1), "BCE+Dice Loss returned NaN"
    assert not torch.isnan(loss2), "Lovasz Loss returned NaN"
    assert loss1.item() >= 0, "BCE+Dice Loss should be non-negative"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Training Loop
    # -------------------------------------------------------------------------
    print("--- Running Training Loop (Demo) ---")

    # Run training for 2 epochs on the mini dataset
    # We use the provided run_fold function
    trained_model = run_fold(
        train_metadata=mini_train_path,
        val_metadata=mini_val_path,
        output_dir=WORK_DIR,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        device=device,
        num_workers=0,  # 0 workers for simple debugging/demo
        base_filters=16,
    )

    # Check if model file was saved
    model_path = os.path.join(WORK_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved after training"

    print(f"Training demo completed successfully. Model saved to {model_path}")

    # -------------------------------------------------------------------------
    # 7. Final Inference Check
    # -------------------------------------------------------------------------
    print("--- Verifying Inference with Saved Model ---")

    # Load model state
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        pred = model(batch_imgs)
        pred_prob = torch.sigmoid(pred)

    assert pred_prob.shape == (1, 1, 128, 128)
    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
