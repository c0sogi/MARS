import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from the provided library
from library.config import Config, AshConfig
from library.dataset import ContrailDataset, get_transforms
from library.model import UNetPlusPlus
from library.losses import CompositeLoss, DiceLoss, FocalLoss
from library.utils import set_seed, rle_encode, GlobalDiceMeter, AverageMeter
from library.train import train_one_epoch, validate


def run_demo():
    print("==== Starting Contrail Identification Library Demo ====")

    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Create a temporary working directory for this demo
    demo_dir = os.path.join(Config.WORKING_DIR, "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Prepare Data Subset (Speed Optimization)
    # --------------------------------------------------------------------------
    # We load the full train metadata but only keep top 16 records for the demo
    # to ensure the training loop finishes in seconds.
    print("\n[1/6] Preparing Data Subset...")
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    subset_df = full_train_df.head(16).copy()

    subset_csv_path = os.path.join(demo_dir, "train_subset.csv")
    subset_df.to_csv(subset_csv_path, index=False)
    print(f"Created subset metadata with {len(subset_df)} records at {subset_csv_path}")

    # 3. Verify Dataset & Preprocessing
    # --------------------------------------------------------------------------
    print("\n[2/6] Verifying Dataset & Ash Composite Normalization...")
    dataset = ContrailDataset(
        metadata_csv_path=subset_csv_path,
        stage="train",
        transform=get_transforms("train"),
    )

    # Fetch one sample
    image, mask = dataset[0]

    # Verify Shapes
    # Image: (3, 256, 256) -> Channels first (ToTensorV2)
    # Mask: (1, 256, 256) -> Channels first
    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {image.shape}"
    assert mask.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected mask shape (1, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {mask.shape}"

    # Verify Normalization (Ash Composite)
    # The dataset normalizes bands to [0, 1].
    # Albumentations ToTensorV2 converts to float tensor but doesn't scale by 255 if input is already float.
    # Our dataset returns float32 in [0, 1].
    min_val, max_val = image.min().item(), image.max().item()
    print(f"Pixel Value Range: [{min_val:.4f}, {max_val:.4f}]")

    # Allow small epsilon for floating point errors or if crop missed 0/1,
    # but generally should be within [0, 1].
    assert (
        min_val >= -0.1 and max_val <= 1.1
    ), "Image data is not properly normalized to approx [0, 1] range."

    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[3/6] Verifying Model Architecture...")
    model = UNetPlusPlus(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # No need to download weights for shape check
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    )
    model.to(device)
    model.eval()

    # Create dummy input batch (B=2, C=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Check output dimensions (B, Classes, H, W)
    assert output.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Model output shape mismatch."

    # 5. Verify Losses & Metrics
    # --------------------------------------------------------------------------
    print("\n[4/6] Verifying Losses and Metrics...")

    # Setup dummy predictions and targets
    # Pred: 0.8 probability (logit ~1.38) at indices 0-10, 0.1 elsewhere
    # Target: 1 at indices 0-10, 0 elsewhere
    # This represents a good prediction.

    # Logits
    logits = torch.full((1, 1, 100, 100), -2.0)  # Low prob
    logits[:, :, 0:10, 0:10] = 2.0  # High prob
    logits = logits.to(device)

    targets = torch.zeros((1, 1, 100, 100))
    targets[:, :, 0:10, 0:10] = 1.0
    targets = targets.to(device)

    # Test Composite Loss
    criterion = CompositeLoss()
    loss = criterion(logits, targets)
    print(f"Calculated Composite Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Test Global Dice Meter
    dice_meter = GlobalDiceMeter()
    dice_meter.update(torch.sigmoid(logits), targets)
    score = dice_meter.get_score()
    print(f"Calculated Dice Score: {score:.4f}")

    # Since predictions match targets closely (logits 2.0 -> sigmoid ~0.88 > 0.5 threshold),
    # Dice should be high (near 1.0).
    assert score > 0.9, f"Dice score should be high for matching masks, got {score}"

    # 6. Verify Utilities (RLE)
    # --------------------------------------------------------------------------
    print("\n[5/6] Verifying RLE Encoding...")
    # Create a simple mask:
    # Row 0: 0 1 1 1 0 ...
    # Flattened (col-major): Pixel 1 is (0,0)=0, Pixel 2 is (1,0)=0...
    # Let's make a 3x3 mask
    # [[0, 1, 0],
    #  [0, 1, 0],
    #  [0, 1, 0]]
    # Flattened Col-Major: 0,0,0 (col0), 1,1,1 (col1), 0,0,0 (col2) -> Indices 4,5,6 (1-based)
    mask_rle = np.zeros((3, 3))
    mask_rle[:, 1] = 1

    encoded = rle_encode(mask_rle)
    print(f"RLE Output: '{encoded}'")

    # Indices 4, 5, 6 are 1s.
    # Run starts at 4, length 3.
    assert encoded == "4 3", f"RLE Encoding incorrect. Expected '4 3', got '{encoded}'"

    # 7. Integration: Training Loop
    # --------------------------------------------------------------------------
    print("\n[6/6] Running Integration Test (Train/Val Loop)...")

    # Loaders
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for demo
        drop_last=True,
    )

    # Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Train 1 Epoch
    print("Running training step...")
    train_loss = train_one_epoch(
        loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        epoch=1,
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Validate
    print("Running validation step...")
    # Re-use train loader as val loader for demo purposes
    val_loss, val_dice = validate(
        loader=train_loader, model=model, criterion=criterion, device=device
    )
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Dice: {val_dice:.4f}")

    assert train_loss > 0, "Train loss should be positive"
    assert val_loss > 0, "Validation loss should be positive"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
