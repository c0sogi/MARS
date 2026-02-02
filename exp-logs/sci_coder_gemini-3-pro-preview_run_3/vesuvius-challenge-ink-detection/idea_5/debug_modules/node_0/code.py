import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encode, fbeta_score
from library.model import DSDN_GN
from library.data import get_loaders
from library.engine import train_one_epoch, evaluate, predict_and_submit, DiceLoss


def run_demo():
    print("=== Vesuvius Ink Detection Demo ===")

    # 1. Configuration Overrides for Speed & Demo
    # -------------------------------------------
    print("[1/7] Configuring environment...")

    # Modify Config to run a fast, lightweight demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # A100 can handle 8 easily
    Config.TTA_ENABLED = False  # Disable Test-Time Augmentation for speed

    # Set demo-specific directories to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = "./working/demo_cache"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set device and seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"      Device: {device}")
    print(f"      Batch Size: {Config.BATCH_SIZE}")
    print(f"      TTA Enabled: {Config.TTA_ENABLED}")

    # 2. Verify Utilities
    # -------------------
    print("[2/7] Verifying utility functions...")

    # Test RLE Encode
    # Create a mask with a single segment: pixels 2 and 3 (1-based indices)
    # Mask: 0 1 1 0
    dummy_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # Flatten: 0, 1, 1, 0
    encoded = rle_encode(dummy_mask)
    # Expected: start at 2, length 2 -> "2 2"
    assert encoded == "2 2", f"RLE Encode failed. Got {encoded}"

    # Test F-Beta Score
    # Perfect match
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])
    score = fbeta_score(y_pred, y_true, beta=0.5)
    assert np.isclose(score, 1.0), f"F-Beta perfect score failed. Got {score}"

    print("      Utilities verified.")

    # 3. Data Loading
    # ---------------
    print("[3/7] Loading data...")
    # load_cached_data=True allows using pre-processed .npy files if available
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Verify loaders are not empty
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Verify batch shapes
    sample_imgs, sample_lbls = next(iter(train_loader))
    # Expected: (B, 65, 256, 256)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    assert (
        sample_imgs.shape == expected_shape
    ), f"Batch shape mismatch. Got {sample_imgs.shape}"
    assert sample_lbls.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Label shape mismatch"

    print(f"      Train Batches: {len(train_loader)}")
    print(f"      Val Batches:   {len(val_loader)}")

    # 4. Model Setup
    # --------------
    print("[4/7] Initializing model...")
    model = DSDN_GN().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        sample_imgs = sample_imgs.to(device)
        output = model(sample_imgs)
        assert output.shape == sample_lbls.shape, "Model output shape mismatch"

    print("      Model initialized and forward pass successful.")

    # 5. Training
    # -----------
    print("[5/7] Training (1 Epoch)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for one epoch using the provided engine function
    # The dataset size is small (Config.BATCH_SIZE * 200 samples), so this is fast
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)

    print(f"      Epoch 1 Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Save checkpoint
    torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # 6. Evaluation
    # -------------
    print("[6/7] Evaluating on Validation Set...")
    # evaluate() reconstructs the full fragments and finds optimal threshold
    val_score, best_threshold = evaluate(model, val_loader, device)

    print(f"      Validation F0.5: {val_score:.4f}")
    print(f"      Optimal Threshold: {best_threshold:.2f}")

    # 7. Inference & Submission
    # -------------------------
    print("[7/7] Generating Submission...")
    # predict_and_submit generates the submission.csv
    predict_and_submit(model, test_loader, device, threshold=best_threshold)

    # Verify Output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission Rows: {len(df_sub)}")
    print(f"      Columns: {list(df_sub.columns)}")

    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Invalid submission format"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
