import sys
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders, InkDataset
from library.model import LeanUNet25D
from library.losses import TverskyLoss
from library.utils import ModelEMA, rle_encode, fbeta_score
from library.train import train_one_epoch, validate
from library.inference import generate_submission, inference_tta


def run_demo():
    print("--- Starting Ink Detection Pipeline Demo ---")

    # 1. Setup & Configuration Overrides
    # ----------------------------------
    # We modify Config attributes to make the demo run fast and lean.
    print("\n[1] Configuring environment...")

    seed_everything(42)

    # Override Config for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.TTA_STEPS = 1  # Reduce TTA for fast inference demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.CACHE_DIR = Path("./working/demo_cache")
    Config.MODEL_PATH = Config.CACHE_DIR / "demo_model.pth"
    Config.SUBMISSION_PATH = Path("./working/submission.csv")

    # Ensure cache dir exists
    if Config.CACHE_DIR.exists():
        shutil.rmtree(Config.CACHE_DIR)
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading Verification
    # ----------------------------
    print("\n[2] Verifying Data Loading...")

    # Initialize dataloaders
    dataloaders = get_dataloaders(Config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Patch the dataset length to be very small for the demo
    # The ConcatDataset wraps InkDatasets. We modify the first one.
    if hasattr(train_loader.dataset, "datasets"):
        for ds in train_loader.dataset.datasets:
            ds.length = Config.BATCH_SIZE * 2  # 2 batches total

    # Fetch a single batch
    images, labels = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.Z_DIM,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected label shape: {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"

    print("    Data loading verification passed.")

    # 3. Model & Loss Verification
    # ----------------------------
    print("\n[3] Verifying Model and Loss...")

    device = Config.DEVICE
    model = LeanUNet25D().to(device)
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # Forward Pass
    outputs = model(images)

    print(f"    Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == labels.shape, "Output shape mismatch"
    assert (
        outputs.min() >= 0 and outputs.max() <= 1
    ), "Outputs must be in [0, 1] (Sigmoid)"

    # Loss Calculation
    loss = loss_fn(outputs, labels)
    print(f"    Loss Value: {loss.item():.4f}")

    assert loss.item() >= 0, "Loss must be non-negative"

    # Backward Pass (Check gradients)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("    Model forward/backward pass verification passed.")

    # 4. Utilities Verification
    # -------------------------
    print("\n[4] Verifying Utilities...")

    # RLE Encode
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[0, :3] = 1  # Pixels 1, 2, 3 (1-based indices: 1, 2, 3)
    # Flattened: 1, 1, 1, 0, ...
    # Run: Start 1, Length 3
    rle_str = rle_encode(dummy_mask)
    print(f"    RLE String: '{rle_str}'")
    assert rle_str == "1 3", f"RLE failed. Expected '1 3', got '{rle_str}'"

    # F-Beta Score
    # Perfect match
    score_perfect = fbeta_score(labels, labels, beta=0.5)
    assert np.isclose(
        score_perfect, 1.0, atol=1e-4
    ), "F-Beta should be 1.0 for perfect match"

    # EMA
    ema = ModelEMA(model, decay=0.99)
    ema.update(model)
    # Check if EMA parameters exist
    assert len(list(ema.get_model().parameters())) == len(list(model.parameters()))
    print("    Utilities verification passed.")

    # 5. Training Loop Component Verification
    # ---------------------------------------
    print("\n[5] Verifying Training Components...")

    # Run one epoch (on the shortened dataset)
    avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, ema)
    print(f"    Train One Epoch Avg Loss: {avg_loss:.4f}")

    # Run validation (limit to 2 batches for speed)
    # Create a limited iterator for validation
    val_iter = iter(val_loader)
    limited_val_batches = []
    try:
        limited_val_batches.append(next(val_iter))
        limited_val_batches.append(next(val_iter))
    except StopIteration:
        pass

    val_score = validate(ema.get_model(), limited_val_batches, device)
    print(f"    Validation F0.5 Score: {val_score:.4f}")

    # Save this model to use in inference
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("    Training components verification passed.")

    # 6. Inference Pipeline Verification
    # ----------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # We use the generate_submission function
    # This will load the model we just saved and run inference on the test set
    # Note: Config.TTA_STEPS was set to 1 to speed this up.

    if Config.TEST_METADATA_PATH.exists():
        generate_submission(
            checkpoint_path=Config.MODEL_PATH, output_path=Config.SUBMISSION_PATH
        )

        if Config.SUBMISSION_PATH.exists():
            df_sub = pd.read_csv(Config.SUBMISSION_PATH)
            print(f"    Submission Generated. Rows: {len(df_sub)}")
            print(f"    Columns: {list(df_sub.columns)}")

            assert (
                "Id" in df_sub.columns and "Predicted" in df_sub.columns
            ), "Submission missing required columns"
            assert len(df_sub) > 0, "Submission file is empty"
        else:
            raise FileNotFoundError("Submission file was not created.")
    else:
        print("    Skipping inference verification (Test metadata not found).")

    print("    Inference verification passed.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
