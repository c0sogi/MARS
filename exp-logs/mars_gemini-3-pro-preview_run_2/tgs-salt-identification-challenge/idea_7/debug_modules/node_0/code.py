import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library import config, utils, dataset, model, losses, train, predict


def main():
    print("=== Starting Demonstration and Verification Script ===")

    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("\n[1] Setting up environment...")
    utils.set_seed(config.SEED)

    # Override config for speed in this demonstration
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print("Configuration configured for fast demonstration.")

    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions (library.utils)...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a 10x10 square of salt

    rle_str = utils.rle_encode(dummy_mask)
    decoded_mask = utils.rle_decode(rle_str, shape=(101, 101))

    assert isinstance(rle_str, str), "RLE encode should return a string"
    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "Decoded mask does not match original mask"
    print(" - RLE Encode/Decode: OK")

    # Test Metric Calculation
    # Case 1: Perfect match
    pred_perfect = np.zeros((1, 101, 101), dtype=np.uint8)
    pred_perfect[0, 10:20, 10:20] = 1
    truth_perfect = np.zeros((1, 101, 101), dtype=np.uint8)
    truth_perfect[0, 10:20, 10:20] = 1

    score_perfect = utils.do_kaggle_metric(pred_perfect, truth_perfect, threshold=0.5)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should have score 1.0, got {score_perfect}"

    # Case 2: No overlap
    pred_wrong = np.zeros((1, 101, 101), dtype=np.uint8)
    pred_wrong[0, 50:60, 50:60] = 1
    score_wrong = utils.do_kaggle_metric(pred_wrong, truth_perfect, threshold=0.5)
    assert np.isclose(
        score_wrong, 0.0
    ), f"No overlap should have score 0.0, got {score_wrong}"

    print(" - Kaggle Metric Calculation: OK")

    # 3. Verify Dataset Loading
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset Loading (library.dataset)...")

    # Load loaders using the library function
    # This handles caching automatically.
    train_loader, val_loader, depth_mean, depth_std = dataset.get_train_val_loaders(
        load_cached_data=True
    )

    print(f" - Train Loader Length: {len(train_loader)}")
    print(f" - Val Loader Length: {len(val_loader)}")
    print(f" - Depth Stats: Mean={depth_mean:.4f}, Std={depth_std:.4f}")

    # Verify Batch Structure
    images, masks, depths = next(iter(train_loader))

    # Check Shapes
    # Config sets target size to 128x128
    assert images.shape == (
        config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (
        config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {masks.shape}"
    assert depths.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Unexpected depth shape: {depths.shape}"

    # Check Data Ranges
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images should be normalized to [0, 1]"
    assert torch.unique(masks).numel() <= 2, "Masks should be binary (0 and 1)"

    print(" - Dataset Batch Structure: OK")

    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture (library.model)...")

    net = model.WideLinkNet34().to(config.DEVICE)

    # Pass the dummy batch from step 3
    images = images.to(config.DEVICE)

    seg_logits, depth_preds = net(images)

    assert seg_logits.shape == (
        config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Output seg shape mismatch: {seg_logits.shape}"
    assert depth_preds.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Output depth shape mismatch: {depth_preds.shape}"

    print(" - Model Forward Pass: OK")

    # 5. Verify Loss Function
    # ---------------------------------------------------------
    print("\n[5] Verifying Loss Function (library.losses)...")

    criterion = losses.SaltNetLoss()

    masks = masks.to(config.DEVICE)
    depths = depths.to(config.DEVICE)

    loss, metrics = criterion(seg_logits, depth_preds, masks, depths)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert "loss_bce" in metrics, "Metrics missing BCE component"
    assert "loss_lovasz" in metrics, "Metrics missing Lovasz component"
    assert "loss_depth" in metrics, "Metrics missing Depth component"

    print(f" - Loss Calculation: OK (Value: {loss.item():.4f})")

    # 6. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[6] Verifying Training Pipeline (library.train)...")

    # Run training in debug mode (uses subset of data, runs for 2 epochs)
    # This validates the integration of dataset, model, and training logic.
    d_mean, d_std = train.run_training(
        epochs=2, batch_size=4, debug=True, device=config.DEVICE
    )

    assert os.path.exists(config.CHECKPOINT_PATH), "Model checkpoint was not saved"
    print(" - Training Loop (Debug Mode): OK")

    # 7. Verify Inference Pipeline
    # ---------------------------------------------------------
    print("\n[7] Verifying Inference Pipeline (library.predict)...")

    # We use the predict module to generate a submission
    # This tests threshold optimization, TTA, cropping, and RLE encoding

    # Ensure we use the stats returned from training
    predict.predict(depth_mean=d_mean, depth_std=d_std)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if IDs match test set
    test_df = pd.read_csv(config.TEST_METADATA_PATH)
    assert len(df_sub) == len(
        test_df
    ), f"Submission row count {len(df_sub)} != Test set size {len(test_df)}"

    print(" - Inference Pipeline: OK")
    print(f" - Submission saved to: {config.SUBMISSION_PATH}")

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")


if __name__ == "__main__":
    main()
