import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, dice_loss, fbeta_score, predict_tiled
from library.model import DCDN
from library.data import get_global_stats, InkDataset, get_dataloaders, load_fragment
from library.train import train_model


def run_demo():
    print("--- Starting End-to-End Demo ---")

    # 1. Configuration Overrides for Demo Speed
    # We override Config attributes to run a minimal version of the task
    Config.WORKING_DIR = Path("./working/demo_execution")
    Config.CHECKPOINT_DIR = Config.WORKING_DIR / "checkpoints"
    Config.PREDICTION_DIR = Config.WORKING_DIR / "predictions"
    Config.CACHE_DIR = Config.WORKING_DIR

    # Reduce compute load
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.THRESHOLD_SEARCH_STEPS = 5  # Reduce steps for threshold optimization

    # Re-run setup to create new directories
    Config.setup()
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Verify Utility Functions
    print("\n[1/5] Verifying Utility Functions...")

    # Test RLE Encoding
    # Pattern: 0 1 1 0 1 0 (Flattened)
    # Indices (1-based): 1 2 3 4 5 6
    # Ink at: 2, 3, 5
    # Runs: Start 2 Len 2, Start 5 Len 1 -> "2 2 5 1"
    dummy_mask = np.array(
        [[0, 1, 1], [0, 1, 0]], dtype=np.uint8
    )  # Flattened: 0 1 1 0 1 0
    encoded = rle_encode(dummy_mask)
    assert (
        encoded == "2 2 5 1"
    ), f"RLE Encoding failed. Expected '2 2 5 1', got '{encoded}'"
    print("  -> RLE Encode: OK")

    # Test Metrics
    dummy_pred = torch.tensor(
        [[[[10.0, -10.0], [-10.0, 10.0]]]]
    )  # Logits: High prob, Low prob, Low prob, High prob
    dummy_target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = dice_loss(dummy_pred, dummy_target)
    score = fbeta_score(dummy_pred, dummy_target, threshold=0.5)

    assert isinstance(loss, torch.Tensor), "Dice loss should return a Tensor"
    assert 0 <= score <= 1, "F-beta score should be between 0 and 1"
    print(f"  -> Metrics (Dice: {loss.item():.4f}, F0.5: {score:.4f}): OK")

    # 3. Verify Model Architecture
    print("\n[2/5] Verifying Model Architecture...")
    model = DCDN().to(Config.DEVICE)

    # Input: (B, Z_DIM, H, W) -> (2, 65, 256, 256)
    dummy_input = torch.randn(
        2, Config.IN_CHANNELS, Config.PATCH_SIZE, Config.PATCH_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print(f"  -> Forward Pass Output Shape {output.shape}: OK")

    # 4. Verify Data Loading
    print("\n[3/5] Verifying Data Loading...")
    # Compute global stats first (this will cache them)
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    mean, std = get_global_stats(df_train, Config.WORKING_DIR, load_cached_data=True)
    print(f"  -> Global Stats: Mean={mean:.2f}, Std={std:.2f}")

    # Initialize Dataset
    train_ds = InkDataset(
        metadata_df=df_train,
        split="train",
        mean=mean,
        std=std,
        num_samples=10,
        cache_dir=Config.WORKING_DIR,
    )

    # Fetch one sample
    vol, label = train_ds[0]
    assert vol.shape == (
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Volume patch shape mismatch"
    assert label.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Label patch shape mismatch"
    print("  -> Dataset Item Fetch: OK")

    # 5. Run Training Loop (Shortened)
    print("\n[4/5] Running Training Loop (1 Epoch, 16 Samples)...")
    # This exercises the data loader, model training step, and the full validation inference
    best_f05 = train_model(load_cached_data=True, num_train_samples=16)
    print(f"  -> Training Finished. Best F0.5: {best_f05:.4f}")

    assert (
        Config.CHECKPOINT_DIR / "best_model.pth"
    ).exists(), "Best model checkpoint not found"
    assert (Config.WORKING_DIR / "threshold.txt").exists(), "Threshold file not found"

    # 6. Run Inference & Generate Submission
    print("\n[5/5] Running Inference on Test Set...")

    # Load best model
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_DIR / "best_model.pth", map_location=Config.DEVICE)
    )
    model.eval()

    # Load threshold
    with open(Config.WORKING_DIR / "threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    print(f"  -> Using Threshold: {threshold:.4f}")

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA)
    submission_rows = []

    for _, row in df_test.iterrows():
        frag_id = row["fragment_id"]
        print(f"  -> Processing Fragment {frag_id}...")

        # Load Volume
        volume, mask, _ = load_fragment(row, Config.WORKING_DIR, load_cached_data=True)

        # Predict
        # Note: predict_tiled handles normalization internally if we pass mean/std,
        # or computes them from the volume if not passed.
        # We should use the training stats for consistency.
        pred_map = predict_tiled(
            model,
            volume,
            patch_size=Config.PATCH_SIZE,
            stride=Config.STRIDE,
            device=Config.DEVICE,
            mean=mean,
            std=std,
        )

        # Apply Mask and Threshold
        # Mask ensures we don't predict ink outside the valid fragment area
        pred_binary = (pred_map > threshold).astype(np.uint8)
        pred_binary = pred_binary * (mask > 0).astype(np.uint8)

        # Encode
        rle_str = rle_encode(pred_binary)
        submission_rows.append({"Id": frag_id, "Predicted": rle_str})

    # Create Submission DataFrame
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv("submission.csv", index=False)

    print("\n--- Demo Complete ---")
    print("Generated submission.csv:")
    print(submission_df.head())


if __name__ == "__main__":
    run_demo()
