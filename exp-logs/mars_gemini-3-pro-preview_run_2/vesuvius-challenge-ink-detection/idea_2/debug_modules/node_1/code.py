import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import rle_encode, fbeta_score, set_seed
from library.dataset import InkDataset
from library.model import InkDetector
from library.losses import BCEDiceLoss
from library.train import train_model
from library.inference import predict_and_submit


def run_demo():
    # --- 1. Configuration & Setup ---
    print(">>> Step 1: Configuring for Demo Run")

    # Modify Config for speed and isolation
    Config.DEBUG = True  # Use a tiny subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.EXP_NAME = "demo_run_script"  # Separate output directory
    Config.BATCH_SIZE = 4  # Small batch size for demonstration

    # Update paths based on new EXP_NAME
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    # Submission path remains ./submission/submission.csv as per config defaults,
    # but we ensure the directory exists via setup.

    # Run setup to create directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # --- 2. Verify Utilities ---
    print("\n>>> Step 2: Verifying Utilities (RLE & Metrics)")

    # Test RLE Encoding
    # Create a simple 3x3 mask:
    # 0 1 0
    # 1 1 1
    # 0 0 0
    # Flattened (row-major): 0, 1, 0, 1, 1, 1, 0, 0, 0
    # Indices (1-based):     1, 2, 3, 4, 5, 6, 7, 8, 9
    # Ink at: 2, 4, 5, 6
    # Runs: Start 2 len 1; Start 4 len 3. -> "2 1 4 3"
    dummy_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    expected_rle = "2 1 4 3"
    assert (
        encoded == expected_rle
    ), f"RLE Encoding failed. Expected '{expected_rle}', got '{encoded}'"
    print("RLE Encoding: OK")

    # Test F-beta Score
    # Perfect match
    preds = torch.tensor([10.0, -10.0, 10.0])  # Logits -> Sigmoid -> ~1, ~0, ~1
    targets = torch.tensor([1.0, 0.0, 1.0])
    score = fbeta_score(torch.sigmoid(preds), targets, beta=0.5, threshold=0.5)
    assert np.isclose(score, 1.0), f"F-beta Score failed for perfect match. Got {score}"
    print("F-beta Score: OK")

    # --- 3. Verify Dataset & Preprocessing ---
    print("\n>>> Step 3: Verifying Dataset & Preprocessing")

    # Load metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            "Metadata not found. Please ensure metadata is generated."
        )

    train_df = pd.read_csv(train_csv_path)

    # Subsample manually for the dataset test to be fast
    sample_df = train_df.head(4)

    # Initialize Dataset
    # This triggers the caching mechanism (loading volume, computing MIP)
    dataset = InkDataset(sample_df, mode="train", load_cached_data=True)

    # Fetch one sample
    image, label, mask = dataset[0]

    # Check Shapes
    # Image: (C, H, W) -> (3, 512, 512)
    assert image.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Image shape mismatch. Expected (3, 512, 512), got {image.shape}"
    # Label: (1, H, W)
    assert label.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Label shape mismatch. Expected (1, 512, 512), got {label.shape}"

    # Check Data Ranges
    assert (
        image.min() >= 0.0 and image.max() <= 1.0
    ), "Image data not normalized to [0, 1]"
    assert (
        label.unique().tolist() == [0.0]
        or label.unique().tolist() == [1.0]
        or set(label.unique().tolist()) == {0.0, 1.0}
    ), "Labels are not binary"

    print("Dataset Loading & Preprocessing: OK")

    # --- 4. Verify Model & Loss ---
    print("\n>>> Step 4: Verifying Model Architecture & Loss")

    model = InkDetector().to(Config.DEVICE)
    criterion = BCEDiceLoss()

    # Create a batch
    batch_images = image.unsqueeze(0).to(Config.DEVICE)  # (1, 3, 512, 512)
    batch_labels = label.unsqueeze(0).to(Config.DEVICE)  # (1, 1, 512, 512)

    # Forward Pass
    logits = model(batch_images)
    assert (
        logits.shape == batch_labels.shape
    ), f"Model output shape mismatch. Expected {batch_labels.shape}, got {logits.shape}"

    # Loss Calculation
    loss = criterion(logits, batch_labels)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print("Model & Loss: OK")

    # Clean up GPU memory
    del model, logits, loss, batch_images, batch_labels
    torch.cuda.empty_cache()

    # --- 5. Run Training Pipeline ---
    print("\n>>> Step 5: Running Training Pipeline (Debug Mode)")

    # train_model() uses Config.DEBUG to subsample data internally
    train_model(load_cached_data=True)

    # Verify artifact generation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to generate model checkpoint at {Config.BEST_MODEL_PATH}"
        )

    print("Training Pipeline: OK")

    # --- 6. Run Inference Pipeline ---
    print("\n>>> Step 6: Running Inference Pipeline")

    # predict_and_submit() loads the best model and generates submission.csv
    predict_and_submit(load_cached_data=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to generate submission file at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission file missing required columns"

    print(f"Submission generated with {len(sub_df)} rows.")
    print("Inference Pipeline: OK")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
