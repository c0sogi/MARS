import os
import numpy as np
import torch
import pandas as pd
import cv2
import warnings

# Import from provided library files
from library import utils
from library import dataset
from library import models
from library import engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of Salt Segmentation pipeline...")

    # 1. Setup
    # Set seed for reproducibility
    utils.np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 2. Demonstrate library.utils
    print("\n--- Testing library.utils ---")

    # Test RLE Encoding/Decoding
    # Create a 101x101 mask with a 10x10 square of 1s in the middle
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[45:55, 45:55] = 1

    rle_str = utils.rle_encode(dummy_mask)
    # Verify RLE string is not empty
    if not rle_str:
        raise AssertionError("RLE encoding returned empty string for non-empty mask.")

    decoded_mask = utils.rle_decode(rle_str, shape=(101, 101))

    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError("RLE Decode does not match original mask!")
    print("RLE Encode/Decode verification passed.")

    # Test IoU
    # Create a second mask slightly shifted
    dummy_mask_2 = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask_2[46:56, 46:56] = 1  # Shifted by 1 pixel

    iou = utils.calc_iou(dummy_mask, dummy_mask_2)
    print(f"Calculated IoU between shifted masks: {iou:.4f}")
    if not (0.0 < iou < 1.0):
        raise AssertionError("IoU calculation seems incorrect for overlapping masks.")

    # Test mAP
    # Perfect match case
    map_score = utils.calc_map([dummy_mask], [dummy_mask])
    if abs(map_score - 1.0) > 1e-6:
        raise AssertionError(f"mAP should be 1.0 for perfect match, got {map_score}")
    print("mAP calculation verification passed.")

    # 3. Demonstrate library.dataset
    print("\n--- Testing library.dataset ---")

    # Prepare data (loads metadata, processes images, caches npy files)
    # This writes to ./working/idea_6/ as defined in library/dataset.py
    print("Loading and processing dataset...")
    # We force load_cached_data=False to demonstrate processing logic,
    # but subsequent runs would use cache.
    data_store = dataset.prepare_data(load_cached_data=False)

    # Verify data keys
    for split in ["train", "val", "test"]:
        if split not in data_store:
            raise AssertionError(f"Missing split {split} in data_store")

    print(f"Train images shape: {data_store['train']['images'].shape}")
    print(f"Train masks shape: {data_store['train']['masks'].shape}")

    # Get DataLoaders
    # Using small batch size for demo speed
    batch_size = 8
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        data_store, batch_size=batch_size, num_workers=2
    )

    # Verify batch structure from Train Loader
    # Expected: image, mask, depth, id
    images, masks, depths, ids = next(iter(train_loader))
    print(
        f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Check dimensions (should be 128x128 due to transforms in dataset.py)
    if images.shape[2:] != (128, 128):
        raise AssertionError(f"Expected image size 128x128, got {images.shape[2:]}")
    if masks.shape[2:] != (128, 128):
        raise AssertionError(f"Expected mask size 128x128, got {masks.shape[2:]}")

    print("Dataset and DataLoader verification passed.")

    # 4. Demonstrate library.models
    print("\n--- Testing library.models ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test DepthRegressor
    regressor = models.DepthRegressor().to(device)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)  # Batch of 2
    reg_out = regressor(dummy_input)

    if reg_out.shape != (2, 1):
        raise AssertionError(
            f"DepthRegressor output shape mismatch. Expected (2, 1), got {reg_out.shape}"
        )
    print("DepthRegressor instantiation and forward pass successful.")

    # Test DepthAwareLinkNet34
    segmenter = models.DepthAwareLinkNet34(num_classes=1).to(device)
    dummy_depth = torch.randn(2, 1).to(device)
    seg_out = segmenter(dummy_input, dummy_depth)

    if seg_out.shape != (2, 1, 128, 128):
        raise AssertionError(
            f"LinkNet output shape mismatch. Expected (2, 1, 128, 128), got {seg_out.shape}"
        )
    print("DepthAwareLinkNet34 instantiation and forward pass successful.")

    # 5. Demonstrate library.engine (Training)
    print("\n--- Testing library.engine (Training) ---")

    # We will train for just 1 epoch to demonstrate functionality and speed.
    # The engine saves checkpoints to ./working/idea_6/

    # A. Regression Training
    print("Running Regression Training (1 Epoch)...")
    reg_model_path = engine.run_regression_training(
        train_loader, val_loader, epochs=1, lr=1e-4, patience=1
    )

    if not os.path.exists(reg_model_path):
        raise AssertionError("Regression model file was not saved.")
    print(f"Regression model saved at: {reg_model_path}")

    # B. Segmentation Training
    print("Running Segmentation Training (1 Epoch)...")
    seg_model_path = engine.run_segmentation_training(
        train_loader, val_loader, epochs=1, lr=1e-3, patience=1
    )

    if not os.path.exists(seg_model_path):
        raise AssertionError("Segmentation model file was not saved.")
    print(f"Segmentation model saved at: {seg_model_path}")

    # 6. Demonstrate Inference and Submission Generation
    print("\n--- Testing Inference and Submission Generation ---")

    # Predict on Test Set
    # Note: engine.predict_segmentation handles TTA and cropping back to 101x101
    print("Running inference on test set...")
    predictions = engine.predict_segmentation(seg_model_path, test_loader)

    print(f"Predictions shape: {predictions.shape}")

    # Expected shape: (N_test, 101, 101)
    if predictions.shape[1:] != (101, 101):
        raise AssertionError(
            f"Prediction spatial dimensions mismatch. Expected 101x101, got {predictions.shape[1:]}"
        )

    if predictions.shape[0] != 1000:
        raise AssertionError(
            f"Prediction count mismatch. Expected 1000, got {predictions.shape[0]}"
        )

    # Generate Submission CSV
    print("Generating submission file...")

    # Binarize predictions (threshold 0.5)
    binary_preds = (predictions > 0.5).astype(np.uint8)

    # Get IDs from test loader to ensure alignment
    test_ids = []
    for _, _, batch_ids in test_loader:
        test_ids.extend(batch_ids)

    if len(test_ids) != len(binary_preds):
        raise AssertionError("Mismatch between number of test IDs and predictions.")

    submission_rows = []
    for idx, mask in zip(test_ids, binary_preds):
        rle = utils.rle_encode(mask)
        submission_rows.append({"id": idx, "rle_mask": rle})

    df_sub = pd.DataFrame(submission_rows)
    output_csv = "./working/submission_demo.csv"
    df_sub.to_csv(output_csv, index=False)

    print(f"Submission file saved to {output_csv}")
    print(f"Head of submission:\n{df_sub.head()}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
