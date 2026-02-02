import os
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import the provided library modules
from library import config, utils, model, dataset, train, inference


def main():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # 1. Setup & Configuration Overrides
    # We override config parameters to ensure the demo runs quickly and uses temporary directories.
    print("\n[1] Configuring environment...")

    # Define temporary directories
    demo_working_dir = "./working/demo_run_script"
    demo_metadata_dir = "./working/demo_metadata"

    # Clean up previous runs if they exist
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_metadata_dir):
        shutil.rmtree(demo_metadata_dir)

    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_metadata_dir, exist_ok=True)

    # Override library.config values
    config.WORKING_DIR = demo_working_dir
    config.METADATA_DIR = demo_metadata_dir
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")
    config.NUM_EPOCHS = 1  # Run only 1 epoch for speed
    config.BATCH_SIZE = 4  # Small batch size
    config.PRETRAINED = False  # Skip downloading heavy weights
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.VALIDATION_THRESHOLD = 0.0  # Force save for demo purposes

    # Set seed for reproducibility
    train.set_seed(config.SEED)
    print("Configuration updated for rapid execution.")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Pattern: 0 1 1 0 0 1 -> Run starts at 2 (len 2), starts at 6 (len 1)
    # 1-based indexing: indices 2,3 are 1s. index 6 is 1.
    dummy_mask = np.array([[0, 1, 1, 0, 0, 1]], dtype=np.uint8)
    rle_output = utils.rle_encoding(dummy_mask)
    expected_rle = "2 2 6 1"
    assert (
        rle_output == expected_rle
    ), f"RLE Encoding failed. Got {rle_output}, expected {expected_rle}"
    print("RLE Encoding verified.")

    # Test Loss Function
    bce_dice_loss = utils.BCEDiceLoss()
    dummy_logits = torch.randn(2, 1, 64, 64)  # (B, C, H, W)
    dummy_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = bce_dice_loss(dummy_logits, dummy_targets)
    assert loss.dim() == 0, "Loss should be a scalar tensor."
    assert not torch.isnan(loss), "Loss contains NaNs."
    print("BCEDiceLoss verified.")

    # Test F-beta Score
    score = utils.fbeta_score(dummy_logits, dummy_targets, beta=0.5)
    assert isinstance(score, float), "F-beta score should return a float."
    assert 0.0 <= score <= 1.0, "F-beta score should be between 0 and 1."
    print("F-beta Score verified.")

    # 3. Data Preparation (Subset Creation)
    print("\n[3] Preparing Data Subsets...")

    # Read original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_val_df = pd.read_csv("./metadata/validation.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Create subsets (take top 4 samples to ensure at least one batch works)
    subset_train_df = orig_train_df.head(4).copy()
    subset_val_df = orig_val_df.head(4).copy()
    # We use the full test df (usually small number of fragments)
    subset_test_df = orig_test_df.copy()

    # Save to temp metadata directory
    subset_train_df.to_csv(os.path.join(demo_metadata_dir, "train.csv"), index=False)
    subset_val_df.to_csv(os.path.join(demo_metadata_dir, "validation.csv"), index=False)
    subset_test_df.to_csv(os.path.join(demo_metadata_dir, "test.csv"), index=False)

    print(
        f"Created subset metadata: Train={len(subset_train_df)}, Val={len(subset_val_df)}"
    )

    # 4. Verify Dataset and Model
    print("\n[4] Verifying Dataset and Model...")

    # Instantiate Dataset
    ds = dataset.InkDataset(
        subset_train_df, mode="train", transforms=dataset.get_transforms("train")
    )
    sample = ds[0]

    # Check keys
    assert "image" in sample and "label" in sample
    # Check shapes
    # Image: (3, H, W) -> 3 channels from MIPs
    # Label: (1, H, W)
    img_shape = sample["image"].shape
    lbl_shape = sample["label"].shape
    assert img_shape == (
        3,
        config.TILE_SIZE,
        config.TILE_SIZE,
    ), f"Unexpected image shape: {img_shape}"
    assert lbl_shape == (
        1,
        config.TILE_SIZE,
        config.TILE_SIZE,
    ), f"Unexpected label shape: {lbl_shape}"
    print("Dataset shapes verified.")

    # Instantiate Model
    # pretrained=False to avoid downloading weights
    seg_model = model.InkSegFormer(pretrained=False)

    # Forward pass check
    # Create a batch of size 2
    batch_imgs = torch.stack([ds[0]["image"], ds[1]["image"]])
    with torch.no_grad():
        outputs = seg_model(batch_imgs)

    assert outputs.shape == (
        2,
        1,
        config.TILE_SIZE,
        config.TILE_SIZE,
    ), f"Model output shape mismatch. Expected (2, 1, {config.TILE_SIZE}, {config.TILE_SIZE}), got {outputs.shape}"
    print("Model forward pass verified.")

    # 5. Run Training Pipeline
    print("\n[5] Running Training Pipeline (1 Epoch)...")

    # This will use the modified config and subset metadata
    train.train_model()

    # Verify artifact generation
    model_path = os.path.join(demo_working_dir, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Training successful. Model saved to {model_path}")
    else:
        # If validation score was 0 (possible with random weights and tiny data), it might not save unless we force it.
        # We set VALIDATION_THRESHOLD to 0.0, so it should save if score > 0.
        # If score is exactly 0, it might not. Let's check if we need to handle that.
        # For demo purposes, we assume it saves or we proceed.
        # If it didn't save, we manually save the model to allow inference step to proceed.
        print(
            "Warning: best_model.pth not found (score likely 0). Saving manual checkpoint for inference demo."
        )
        torch.save(seg_model.state_dict(), model_path)

    # 6. Run Inference Pipeline
    print("\n[6] Running Inference Pipeline...")

    # This will load the model from demo_working_dir and predict on test.csv
    inference.run_inference()

    # Verify submission file
    submission_file = config.SUBMISSION_PATH
    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"Inference successful. Submission generated at {submission_file}")
        print("Submission Head:")
        print(df_sub.head())

        # Validate submission format
        assert (
            "Id" in df_sub.columns and "Predicted" in df_sub.columns
        ), "Submission columns missing."
        assert len(df_sub) == len(subset_test_df), "Submission row count mismatch."
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
