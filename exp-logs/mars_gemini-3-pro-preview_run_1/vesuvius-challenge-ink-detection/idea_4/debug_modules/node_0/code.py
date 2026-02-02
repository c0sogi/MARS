import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib


def run_demo():
    print("=== Starting Vesuvius Challenge Library Demo ===")

    # Ensure reproducibility
    config.set_seed(42)

    # Define temporary paths for the demo
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    DEMO_TRAIN_CSV = os.path.join(DEMO_DIR, "train_mini.csv")
    DEMO_VAL_CSV = os.path.join(DEMO_DIR, "val_mini.csv")
    DEMO_TEST_CSV = os.path.join(DEMO_DIR, "test_mini.csv")
    DEMO_SUBMISSION = os.path.join(DEMO_DIR, "submission_demo.csv")

    # =========================================================================
    # 1. Utilities Demonstration
    # =========================================================================
    print("\n[1/6] Verifying Utilities...")

    # Test RLE Encode
    # Pattern: 0 1 1 1 0 0 1 0 -> Indices (1-based): 2,3,4 are 1s (start 2, len 3), 7 is 1 (start 7, len 1)
    # Flattened logic in utils.rle_encode handles 2D arrays row-major.
    dummy_mask = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.uint8)
    # Flattened: 0 1 1 1 0 0 1 0 0
    # Indices:   1 2 3 4 5 6 7 8 9
    # 1s at: 2, 3, 4, 7
    # Runs: Start 2, Len 3 (2,3,4); Start 7, Len 1 (7)
    # Expected string: "2 3 7 1"
    rle_output = utils.rle_encode(dummy_mask)
    assert (
        rle_output == "2 3 7 1"
    ), f"RLE Encode failed. Expected '2 3 7 1', got '{rle_output}'"
    print("  - RLE Encode: OK")

    # Test F-beta Score
    # Pred: 1 1 0 0
    # True: 1 0 1 0
    # TP=1 (idx 0), FP=1 (idx 1), FN=1 (idx 2)
    # Precision = 1/(1+1) = 0.5
    # Recall = 1/(1+1) = 0.5
    # Beta=0.5 -> Weight precision higher.
    # F0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.array([1, 0, 1, 0])
    fbeta = utils.calculate_fbeta(y_pred, y_true, beta=0.5)
    assert np.isclose(
        fbeta, 0.5
    ), f"F-beta calculation failed. Expected 0.5, got {fbeta}"
    print("  - F-beta Calculation: OK")

    # =========================================================================
    # 2. Model Architecture Demonstration
    # =========================================================================
    print("\n[2/6] Verifying Model Architecture...")

    net = model_lib.ParallelDilatedCNN().to(config.DEVICE)
    # Input shape: (Batch, Channels=65, H=512, W=512)
    # We use a smaller spatial size for the demo forward pass to save time,
    # as the model is fully convolutional and handles arbitrary spatial dims (min 8x8 due to dilations).
    # However, standard patch size is 512. We'll use 128 for speed test.
    dummy_input = torch.randn(2, 65, 128, 128).to(config.DEVICE)

    with torch.no_grad():
        output = net(dummy_input)

    # Output shape should be (Batch, 1, H, W)
    expected_shape = (2, 1, 128, 128)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print(f"  - Forward Pass: OK (Output shape: {output.shape})")

    # =========================================================================
    # 3. Data Preparation (Mini Subsets)
    # =========================================================================
    print("\n[3/6] Preparing Mini Datasets...")

    # Load original metadata
    df_train_full = pd.read_csv(config.TRAIN_METADATA)
    df_val_full = pd.read_csv(config.VAL_METADATA)
    df_test_full = pd.read_csv(config.TEST_METADATA)

    # Create subsets (2 samples each)
    df_train_mini = df_train_full.head(2).copy()
    df_val_mini = df_val_full.head(2).copy()
    df_test_mini = df_test_full.head(2).copy()

    # Save mini metadata
    df_train_mini.to_csv(DEMO_TRAIN_CSV, index=False)
    df_val_mini.to_csv(DEMO_VAL_CSV, index=False)
    df_test_mini.to_csv(DEMO_TEST_CSV, index=False)

    print(
        f"  - Created mini train ({len(df_train_mini)}), val ({len(df_val_mini)}), test ({len(df_test_mini)})"
    )

    # Monkey-patch the dataset module to use these new files
    # This is necessary because get_dataloaders reads the global variables from config/dataset imports
    dataset.TRAIN_METADATA = DEMO_TRAIN_CSV
    dataset.VAL_METADATA = DEMO_VAL_CSV
    dataset.TEST_METADATA = DEMO_TEST_CSV

    print("  - Patched dataset module paths.")

    # =========================================================================
    # 4. Dataset & DataLoader Demonstration
    # =========================================================================
    print("\n[4/6] Verifying Dataset Loading...")

    # Initialize dataset directly
    ds = dataset.InkDataset(
        df_train_mini, data_dir=config.INPUT_DIR, load_cached_data=True
    )

    # Check __getitem__
    vol, mask, sample_id = ds[0]

    # Validate shapes
    # Volume: (65, 512, 512), Mask: (1, 512, 512)
    assert vol.shape == (65, 512, 512), f"Volume shape incorrect: {vol.shape}"
    assert mask.shape == (1, 512, 512), f"Mask shape incorrect: {mask.shape}"
    assert isinstance(vol, torch.Tensor), "Volume is not a tensor"
    print("  - Dataset __getitem__: OK")

    # Check DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=2,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
        load_cached_data=True,
    )

    batch_vol, batch_mask, _ = next(iter(train_loader))
    assert batch_vol.shape[0] == 2, "Batch size mismatch"
    print("  - DataLoader Iteration: OK")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n[5/6] Running Training Demo (1 Epoch)...")

    # We use a small number of epochs and batch size
    # Note: run_training saves to config.CHECKPOINT_DIR.
    # We rely on the default path "./working/idea_4/checkpoints" which is writable.

    best_score = train_lib.run_training(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        pos_weight_val=1.0,
        patience=1,
        load_cached_data=True,
    )

    # Check if checkpoint was created
    expected_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Best model checkpoint was not created."
    print(f"  - Training complete. Best Score: {best_score:.4f}")
    print(f"  - Checkpoint saved to: {expected_ckpt}")

    # =========================================================================
    # 6. Inference Demonstration
    # =========================================================================
    print("\n[6/6] Running Inference Demo...")

    # Run inference using the checkpoint we just created
    inference_lib.predict_and_submit(
        checkpoint_path=expected_ckpt,
        submission_path=DEMO_SUBMISSION,
        device=config.DEVICE,
        batch_size=2,
        num_workers=0,
        load_cached_data=True,
    )

    assert os.path.exists(DEMO_SUBMISSION), "Submission file was not created."

    # Validate submission content
    sub_df = pd.read_csv(DEMO_SUBMISSION)
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    print(f"  - Submission generated at: {DEMO_SUBMISSION}")
    print("  - Content Preview:")
    print(sub_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
