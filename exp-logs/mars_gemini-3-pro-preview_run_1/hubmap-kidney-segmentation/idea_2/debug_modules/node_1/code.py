import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


# --- 1. Suppress TQDM Progress Bars ---
# This must be done before importing library modules that utilize tqdm.
class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return len(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


import tqdm

# Monkeypatch tqdm to be silent for cleaner output
tqdm.tqdm = SilentTqdm
tqdm.auto = type("obj", (object,), {"tqdm": SilentTqdm})
sys.modules["tqdm"] = tqdm
sys.modules["tqdm.auto"] = tqdm.auto

# --- 2. Import Library Modules ---
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, get_device
from library.data import HubmapDataset
from library.model import build_model
from library.train_eval import train_one_epoch, validate, BCEDiceLoss

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    print("Starting HuBMAP Library Verification Script...")

    # --- Configuration Override for Speed ---
    # We modify the Config class directly to enable a fast debug run.
    print("Configuring for fast execution (Debug Mode)...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 8  # Process only 8 tiles per dataset
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = (
        0  # Disable multiprocessing to avoid overhead in this short script
    )

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Computation Device: {device}")

    # --- 4. Verify Utils (RLE Logic) ---
    print("\n--- Verifying Utils (RLE Encoding/Decoding) ---")
    # Create a synthetic 4x4 binary mask
    # Pattern:
    # 0 1 0 0
    # 1 1 0 0
    # 0 0 0 0
    # 0 0 1 0
    mask_dummy = np.zeros((4, 4), dtype=np.uint8)
    mask_dummy[1, 0] = 1
    mask_dummy[0, 1] = 1
    mask_dummy[1, 1] = 1
    mask_dummy[3, 2] = 1

    # Encode and then Decode
    encoded = rle_encode(mask_dummy)
    decoded = rle_decode(encoded, (4, 4))

    # Verify integrity
    if not np.array_equal(mask_dummy, decoded):
        raise AssertionError(
            "RLE Encode/Decode mismatch! The decoded mask does not match the original."
        )
    print(f"RLE Check Passed. Encoded string: '{encoded}'")

    # --- 5. Verify Data Loading ---
    print("\n--- Verifying Data Loading Pipeline ---")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Optimization: Limit dataframe to 1 image to speed up mask preprocessing (npy conversion)
    df_train_subset = df_train.head(1).copy()

    # Initialize Dataset in Train mode
    # load_cached_data=False forces the tile generation logic to run
    train_dataset = HubmapDataset(df_train_subset, mode="train", load_cached_data=False)

    print(
        f"Dataset initialized. Number of samples (Debug limited): {len(train_dataset)}"
    )
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty! Check input data or debug constraints.")

    # Fetch one sample to verify structure
    sample = train_dataset[0]
    image = sample["image"]
    mask = sample["mask"]

    # Assertions on Tensor Shapes
    expected_size = Config.TILE_SIZE
    if image.shape != (3, expected_size, expected_size):
        raise AssertionError(
            f"Image shape mismatch. Expected (3, {expected_size}, {expected_size}), got {image.shape}"
        )

    if mask.shape != (1, expected_size, expected_size):
        raise AssertionError(
            f"Mask shape mismatch. Expected (1, {expected_size}, {expected_size}), got {mask.shape}"
        )

    print(
        f"Sample loaded successfully. Image Tensor: {image.shape}, Mask Tensor: {mask.shape}"
    )

    # --- 6. Verify Model Architecture ---
    print("\n--- Verifying Model Architecture ---")
    model = build_model()
    model.to(device)

    # Run a dummy forward pass to check connectivity and output dimensions
    dummy_input = torch.randn(2, 3, expected_size, expected_size).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (2, 1, expected_size, expected_size):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1, {expected_size}, {expected_size}), got {output.shape}"
        )

    print("Model forward pass successful.")

    # --- 7. Verify Training Loop Components ---
    print("\n--- Verifying Training & Validation Loop ---")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Setup standard training components
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = BCEDiceLoss()
    scaler = GradScaler()

    # Run one epoch of training (on the small debug subset)
    print("Executing training step...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, device, scaler
    )
    print(f"Train Step Complete. Loss: {train_loss:.4f}")

    # Prepare Validation Set
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_val_subset = df_val.head(1).copy()  # Limit to 1 image for speed
    val_dataset = HubmapDataset(df_val_subset, mode="val", load_cached_data=False)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Run validation
    print("Executing validation step...")
    # Note: Dice score might be low/zero because we only process a few tiles in Debug mode,
    # leaving the rest of the reconstructed image empty.
    val_loss, val_dice = validate(model, val_loader, df_val_subset, device, criterion)

    print(f"Validation Step Complete. Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    # --- 8. Verify Inference Logic ---
    print("\n--- Verifying Inference/TTA Logic ---")
    model.eval()

    # Simulate inference on a single batch from the validation loader
    batch = next(iter(val_loader))
    images = batch["image"].to(device)

    with torch.no_grad():
        # Simulate Test-Time Augmentation (TTA) - Horizontal Flip
        # 1. Standard Forward Pass
        pred_1 = torch.sigmoid(model(images))

        # 2. Horizontal Flip Forward Pass
        images_flipped = torch.flip(images, dims=[3])
        pred_2 = torch.sigmoid(model(images_flipped))
        pred_2 = torch.flip(pred_2, dims=[3])  # Flip back

        # Average predictions
        avg_pred = (pred_1 + pred_2) / 2.0

    if avg_pred.shape != (Config.BATCH_SIZE, 1, expected_size, expected_size):
        raise AssertionError("Inference output shape mismatch")

    print("Inference logic verified successfully.")

    print("\nAll verification steps completed. The library is functional.")
