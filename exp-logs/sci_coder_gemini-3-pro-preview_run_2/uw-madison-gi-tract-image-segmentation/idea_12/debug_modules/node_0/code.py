import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_dice, calculate_hausdorff
from library.data_loader import get_dataloaders
from library.models import GhostUNet, EfficientNetUNet
from library.losses import BCEDiceLoss, TverskyLoss
from library.trainer import Trainer
from library.inference import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Define a working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config to use demo settings
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.COARSE_MODEL_PATH = os.path.join(DEMO_DIR, "coarse_model.pth")
    Config.FINE_MODEL_PATH = os.path.join(DEMO_DIR, "fine_model.pth")

    # Enable Debug mode to use small data subsets
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small sample for speed

    # Reduce training parameters
    Config.COARSE_EPOCHS = 1
    Config.FINE_EPOCHS = 1
    Config.COARSE_BATCH_SIZE = 4
    Config.FINE_BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Create a mini test metadata file for fast inference
    full_test_meta = pd.read_csv(Config.TEST_META_PATH)
    # Take a small number of unique slices (3 rows per slice usually)
    mini_test_meta = full_test_meta.head(30).copy()
    mini_test_path = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
    mini_test_meta.to_csv(mini_test_path, index=False)
    Config.TEST_META_PATH = mini_test_path

    set_seed(Config.SEED)
    print("Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Get Coarse Dataloaders
    train_loader, val_loader = get_dataloaders(
        stage="coarse", batch_size=Config.COARSE_BATCH_SIZE
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    print(f"Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}")

    # Assertions
    # Images: (B, 3, H, W) -> 2.5D stack has 3 channels
    assert images.ndim == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (2.5D stack)"
    assert (
        images.shape[2:] == Config.COARSE_IMG_SIZE
    ), f"Image size mismatch. Expected {Config.COARSE_IMG_SIZE}"

    # Masks: (B, 3, H, W) -> 3 classes
    assert masks.ndim == 4, "Masks should be 4D tensors"
    assert (
        masks.shape[1] == Config.NUM_CLASSES
    ), f"Masks should have {Config.NUM_CLASSES} channels"

    # Check Normalization
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"
    assert set(torch.unique(masks).numpy()).issubset(
        {0.0, 1.0}
    ), "Masks should be binary (0 or 1)"

    print("Data Pipeline verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architectures...")

    device = Config.DEVICE

    # Test Coarse Model (GhostUNet)
    coarse_model = GhostUNet(in_channels=3, num_classes=3).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = coarse_model(dummy_input)

    print(f"GhostUNet Output Shape: {output.shape}")
    assert output.shape == (2, 3, 256, 256), "GhostUNet output shape mismatch"

    # Test Fine Model (EfficientNetUNet)
    fine_model = EfficientNetUNet(in_channels=3, num_classes=3).to(device)
    dummy_input_fine = torch.randn(2, 3, 320, 320).to(device)
    with torch.no_grad():
        output_fine = fine_model(dummy_input_fine)

    print(f"EfficientNetUNet Output Shape: {output_fine.shape}")
    assert output_fine.shape == (
        2,
        3,
        320,
        320,
    ), "EfficientNetUNet output shape mismatch"

    print("Models verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss Functions...")

    # Setup dummy data
    pred_logits = torch.randn(4, 3, 256, 256)
    target_masks = torch.randint(0, 2, (4, 3, 256, 256)).float()

    # BCEDiceLoss
    bce_dice = BCEDiceLoss()
    loss_val = bce_dice(pred_logits, target_masks)
    print(f"BCEDiceLoss Value: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "BCEDiceLoss returned NaN"
    assert loss_val.item() >= 0, "BCEDiceLoss should be non-negative"

    # TverskyLoss
    tversky = TverskyLoss()
    loss_val_t = tversky(pred_logits, target_masks)
    print(f"TverskyLoss Value: {loss_val_t.item():.4f}")
    assert not torch.isnan(loss_val_t), "TverskyLoss returned NaN"
    assert loss_val_t.item() >= 0, "TverskyLoss should be non-negative"

    print("Loss functions verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch per stage)...")

    # Train Coarse Model
    print(">> Training Stage 1: Coarse Model")
    coarse_trainer = Trainer(stage="coarse")
    coarse_trainer.fit()

    assert os.path.exists(Config.COARSE_MODEL_PATH), "Coarse model checkpoint not saved"

    # Train Fine Model
    print(">> Training Stage 2: Fine Model")
    fine_trainer = Trainer(stage="fine")
    fine_trainer.fit()

    assert os.path.exists(Config.FINE_MODEL_PATH), "Fine model checkpoint not saved"

    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Mini Test Set...")

    inference_engine = InferenceEngine()
    inference_engine.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["id", "class", "predicted"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check if we have predictions (some might be empty strings if background, which is fine)
    # Just ensure the column exists and is string-like or NaN
    assert "predicted" in sub_df.columns

    print("Inference and submission generation verified.")

    # -------------------------------------------------------------------------
    # 7. Metric Utilities Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Metric Utilities...")

    # Create synthetic 3D volumes (Depth, Height, Width)
    # Case 1: Perfect overlap
    vol_true = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_true[2:4, 40:60, 40:60] = 1

    vol_pred = vol_true.copy()

    dice = calculate_dice(vol_true, vol_pred)
    hd = calculate_hausdorff(vol_true, vol_pred)

    print(f"Perfect Overlap -> Dice: {dice:.4f}, Hausdorff: {hd:.4f}")
    assert np.isclose(dice, 1.0), "Dice should be 1.0 for perfect overlap"
    assert np.isclose(hd, 0.0), "Hausdorff should be 0.0 for perfect overlap"

    # Case 2: Offset prediction
    vol_pred_offset = np.zeros_like(vol_true)
    # Shift by 10 pixels
    vol_pred_offset[2:4, 50:70, 50:70] = 1

    dice_off = calculate_dice(vol_true, vol_pred_offset)
    hd_off = calculate_hausdorff(vol_true, vol_pred_offset)

    print(f"Offset Overlap -> Dice: {dice_off:.4f}, Hausdorff: {hd_off:.4f}")
    assert dice_off < 1.0, "Dice should be < 1.0 for offset"
    assert hd_off > 0.0, "Hausdorff should be > 0.0 for offset"

    print("Metrics verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
