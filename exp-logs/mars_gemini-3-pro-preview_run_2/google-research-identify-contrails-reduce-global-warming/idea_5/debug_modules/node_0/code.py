import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
from library.config import Config
from library.utils import set_seed, rle_encode, metric_global_dice
from library.dataset import ContrailsDataset
from library.model import ContextEnhancedUNet
from library.loss import HybridBatchDiceLoss
from library.training import train_model
from library.inference import predict_and_submit


def run_demo():
    print("=== Starting Contrail Identification Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("[1] Setting up configuration for fast execution...")

    # Override Config paths to use a demo working directory
    DEMO_WORK_DIR = "./working/demo_run"
    DEMO_SUB_DIR = "./working/demo_submission"

    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    if os.path.exists(DEMO_SUB_DIR):
        shutil.rmtree(DEMO_SUB_DIR)
    os.makedirs(DEMO_SUB_DIR, exist_ok=True)

    # Monkey-patch Config attributes
    Config.WORKING_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_DIR = DEMO_SUB_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_WORK_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUB_DIR, "submission.csv")

    # Reduce compute requirements for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use main thread to avoid overhead
    Config.DEBUG_SAMPLE_SIZE = 12
    Config.IMG_SIZE = 256

    # Set seed
    set_seed(Config.SEED)
    print("    Configuration updated. Random seed set.\n")

    # ---------------------------------------------------------
    # 2. Verify Utilities (RLE & Metric)
    # ---------------------------------------------------------
    print("[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a 3x3 mask:
    # [[0, 0, 0],
    #  [1, 0, 0],
    #  [0, 0, 0]]
    # Column-major flatten: 0, 1, 0 (col 0) -> 0, 0, 0 (col 1) -> 0, 0, 0 (col 2)
    # 1-based indices: Index 2 is 1. Run: Start 2, Length 1.
    dummy_mask = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    rle_result = rle_encode(dummy_mask)
    assert rle_result == "2 1", f"RLE failed. Expected '2 1', got '{rle_result}'"
    print("    RLE Encoding: OK")

    # Test Global Dice Metric
    # Case 1: Perfect Match
    dice_perfect = metric_global_dice(dummy_mask, dummy_mask)
    assert np.isclose(dice_perfect, 1.0), f"Dice Perfect failed: {dice_perfect}"

    # Case 2: No Overlap
    dummy_pred = np.zeros_like(dummy_mask)
    dice_zero = metric_global_dice(dummy_pred, dummy_mask)
    assert np.isclose(dice_zero, 0.0), f"Dice Zero failed: {dice_zero}"

    print("    Global Dice Metric: OK\n")

    # ---------------------------------------------------------
    # 3. Verify Dataset Loading
    # ---------------------------------------------------------
    print("[3] Verifying Dataset...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Take a small subset
    subset_df = train_df.head(4).copy()

    # Instantiate Dataset
    dataset = ContrailsDataset(subset_df, train=True)

    # Fetch one sample
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    record_id = sample["record_id"]

    # Verify Shapes
    # Image: (6, 256, 256), Mask: (1, 256, 256)
    assert image.shape == (6, 256, 256), f"Image shape mismatch: {image.shape}"
    assert mask.shape == (1, 256, 256), f"Mask shape mismatch: {mask.shape}"
    assert isinstance(image, torch.Tensor), "Image is not a tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a tensor"

    print(f"    Loaded Record ID: {record_id}")
    print(f"    Image Shape: {image.shape}")
    print(f"    Mask Shape: {mask.shape}")
    print("    Dataset Loading: OK\n")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("[4] Verifying Model Architecture...")

    model = ContextEnhancedUNet()
    model.eval()

    # Create dummy input batch: (Batch=2, Channels=6, H=256, W=256)
    dummy_input = torch.randn(2, 6, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (2, 1, 256, 256)
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch: {output.shape}"

    print("    Forward Pass: OK")
    print(f"    Output Shape: {output.shape}\n")

    # ---------------------------------------------------------
    # 5. Verify Loss Function
    # ---------------------------------------------------------
    print("[5] Verifying Loss Function...")

    criterion = HybridBatchDiceLoss()

    # Logits (B, 1, H, W) and Targets (B, 1, H, W)
    dummy_logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

    loss = criterion(dummy_logits, dummy_targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Check backward pass capability
    loss.backward()
    assert dummy_logits.grad is not None, "Gradients not computed"

    print(f"    Calculated Loss: {loss.item():.4f}")
    print("    Loss Function: OK\n")

    # ---------------------------------------------------------
    # 6. Run Training Loop (Debug Mode)
    # ---------------------------------------------------------
    print("[6] Running Training Loop (Debug Mode)...")

    # This will train for 1 epoch on Config.DEBUG_SAMPLE_SIZE samples
    # and save the model to Config.BEST_MODEL_PATH
    train_model(debug=True)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print(f"    Model saved to: {Config.BEST_MODEL_PATH}")
    print("    Training Loop: OK\n")

    # ---------------------------------------------------------
    # 7. Run Inference Pipeline
    # ---------------------------------------------------------
    print("[7] Running Inference Pipeline...")

    # Create a dummy test metadata file for the demo if needed,
    # but here we use the existing test_metadata.csv with debug flag
    # which subsets the data inside predict_and_submit.

    predict_and_submit(
        model_path=Config.BEST_MODEL_PATH,
        metadata_path=Config.TEST_METADATA_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "record_id" in sub_df.columns, "Missing record_id column"
    assert "encoded_pixels" in sub_df.columns, "Missing encoded_pixels column"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"    Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"    Rows generated: {len(sub_df)}")
    print("    Inference Pipeline: OK\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
