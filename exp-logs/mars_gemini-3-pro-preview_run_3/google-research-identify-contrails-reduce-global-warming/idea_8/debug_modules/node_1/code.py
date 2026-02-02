import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import warnings

# Import library modules
from library.config import Config
from library.dataset import ContrailDataset
from library.model import ContrailUNet
from library.losses import DiceBCELoss
from library.utils import seed_everything, dice_coef, rle_encode
from library.train import train_fn, valid_fn, CheckpointManager
from library.inference import predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("=== Starting Contrail Segmentation Library Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # We override Config parameters to run a fast demo (mini-batch, few epochs, subset data)
    print("\n[1] Configuring environment for demonstration...")

    # Define paths for demo
    DEMO_DIR = os.path.join(os.getcwd(), "working", "demo_execution")
    DEMO_METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    DEMO_PREDICTIONS_DIR = os.path.join(DEMO_DIR, "predictions")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEMO_PREDICTIONS_DIR, exist_ok=True)

    # Patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    Config.PREDICTIONS_DIR = DEMO_PREDICTIONS_DIR
    Config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.TOP_K_CHECKPOINTS = 1

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Prepare Subset Metadata
    # ---------------------------------------------------------
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/validation.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (take top 20 samples)
    subset_size = 20
    train_subset = orig_train.head(subset_size).copy()
    val_subset = orig_val.head(subset_size).copy()
    test_subset = orig_test.head(subset_size).copy()

    # Save subset metadata
    train_meta_path = os.path.join(DEMO_METADATA_DIR, "train_subset.csv")
    val_meta_path = os.path.join(DEMO_METADATA_DIR, "val_subset.csv")
    test_meta_path = os.path.join(DEMO_METADATA_DIR, "test_subset.csv")

    train_subset.to_csv(train_meta_path, index=False)
    val_subset.to_csv(val_meta_path, index=False)
    test_subset.to_csv(test_meta_path, index=False)

    # Update Config to point to subsets
    Config.TRAIN_METADATA_PATH = train_meta_path
    Config.VALIDATION_METADATA_PATH = val_meta_path
    Config.TEST_METADATA_PATH = test_meta_path

    print(f"    Created subsets with {subset_size} samples each.")

    # ---------------------------------------------------------
    # 3. Verify Dataset & Transforms
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset class...")

    ds = ContrailDataset(metadata_path=Config.TRAIN_METADATA_PATH, split="train")
    print(f"    Dataset length: {len(ds)}")

    # Fetch one sample
    img_tensor, mask_tensor = ds[0]

    # Verify shapes
    # Input: (9 channels, H, W) -> 3 timestamps * 3 features (Ash, Vel, Accel)
    # Mask: (1 channel, H, W)
    print(f"    Input Tensor Shape: {img_tensor.shape}")
    print(f"    Mask Tensor Shape: {mask_tensor.shape}")

    assert img_tensor.shape == (
        9,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected input shape (9, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected mask shape (1, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {mask_tensor.shape}"

    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = ContrailUNet().to(device)

    # Create a dummy batch
    dummy_input = img_tensor.unsqueeze(0).to(device)  # (1, 9, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    assert output.shape == (
        1,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected output shape (1, 1, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {output.shape}"

    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 5. Verify Loss and Metric
    # ---------------------------------------------------------
    print("\n[5] Verifying Loss and Metric...")

    loss_fn = DiceBCELoss()
    dummy_target = mask_tensor.unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Calculate Loss
    loss = loss_fn(output, dummy_target)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Calculate Metric
    metric = dice_coef(output, dummy_target)
    print(f"    Calculated Dice: {metric:.4f}")
    assert 0 <= metric <= 1, "Dice score out of range [0, 1]"

    print("    Loss and Metric verification passed.")

    # ---------------------------------------------------------
    # 6. Execute Training Loop (1 Epoch)
    # ---------------------------------------------------------
    print("\n[6] Executing Training Loop (1 Epoch on subset)...")

    # Setup DataLoaders
    train_loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    val_ds = ContrailDataset(
        metadata_path=Config.VALIDATION_METADATA_PATH, split="validation"
    )
    valid_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Setup Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()
    ckpt_manager = CheckpointManager(
        Config.CHECKPOINT_DIR, top_k=Config.TOP_K_CHECKPOINTS
    )

    # Run Train Step
    train_loss = train_fn(
        model, train_loader, optimizer, loss_fn, scaler, device, epoch=1
    )

    # Run Valid Step
    val_dice = valid_fn(model, valid_loader, device)

    print(
        f"    Epoch 1 Result - Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}"
    )

    # Save Checkpoint
    ckpt_manager.save(model, 1, val_dice)

    # Verify checkpoint exists
    saved_ckpts = ckpt_manager.get_best_checkpoints()
    assert len(saved_ckpts) > 0, "Checkpoint was not saved."
    assert os.path.exists(
        saved_ckpts[0]
    ), f"Checkpoint file missing at {saved_ckpts[0]}"

    print("    Training loop verification passed.")

    # ---------------------------------------------------------
    # 7. Execute Inference
    # ---------------------------------------------------------
    print("\n[7] Executing Inference on Test Subset...")

    # The predict function loads checkpoints from Config.CHECKPOINT_DIR
    # We just saved one there.

    try:
        predict()
    except Exception as e:
        print(f"    Inference failed with error: {e}")
        raise e

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission generated with {len(sub_df)} rows.")
        print(f"    Columns: {list(sub_df.columns)}")

        assert len(sub_df) == len(
            test_subset
        ), f"Submission rows ({len(sub_df)}) do not match test subset size ({len(test_subset)})"
        assert (
            "record_id" in sub_df.columns and "encoded_pixels" in sub_df.columns
        ), "Submission missing required columns"

        # Check if RLE format looks correct (either '-' or numbers)
        sample_rle = sub_df.iloc[0]["encoded_pixels"]
        is_valid_rle = (sample_rle == "-") or (
            all(x.isdigit() for x in sample_rle.split())
        )
        assert is_valid_rle, f"Invalid RLE format detected: {sample_rle}"

        print("    Inference verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Complete Successfully ===")
