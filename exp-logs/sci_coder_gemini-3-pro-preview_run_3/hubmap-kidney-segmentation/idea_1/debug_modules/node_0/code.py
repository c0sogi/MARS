import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

from library.config import Config, seed_everything
from library.utils import rle_encode, rle_decode, polygons_to_mask, dice_coef
from library.dataset import HuBMAPDataset
from library.model import FPNResNet18
from library.losses import BCEDiceLoss
from library.train import run_training
from library.inference import generate_submission_csv


def clean_artifacts():
    """Clean up previous artifacts to ensure fresh run."""
    if os.path.exists(Config.ARTIFACT_DIR):
        shutil.rmtree(Config.ARTIFACT_DIR)
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)


def verify_utils():
    print("\n--- Verifying Utils ---")
    # 1. RLE Encoding/Decoding
    shape = (256, 256)
    mask = np.zeros(shape, dtype=np.uint8)
    # Create a square
    mask[50:100, 50:100] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape)

    if not np.array_equal(mask, decoded):
        raise AssertionError("RLE Encode/Decode roundtrip failed.")
    print("RLE Encode/Decode: Passed")

    # 2. Polygons to Mask
    # Triangle
    poly = [[[10, 10], [10, 50], [50, 10]]]
    p_mask = polygons_to_mask(poly, (100, 100))
    if p_mask.sum() == 0:
        raise AssertionError("Polygon to mask failed: Empty mask.")
    print("Polygons to Mask: Passed")

    # 3. Dice Coefficient
    y_t = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    y_p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    # Intersection = 1, Sum = 2+1=3. Dice = 2*1 / (3) = 0.66...
    dice = dice_coef(y_t, y_p, smooth=0.0)
    if abs(dice.item() - (2.0 / 3.0)) > 1e-4:
        raise AssertionError(
            f"Dice Coefficient calculation incorrect. Got {dice.item()}"
        )
    print("Dice Coefficient: Passed")


def verify_dataset():
    print("\n--- Verifying Dataset ---")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    # Use only 1 image for speed
    train_df_subset = train_df.head(1)

    # Initialize dataset
    # load_cached_data=False to force tile generation logic verification
    # tissue_overlap_threshold=0.0 to ensure we get tiles even if tissue is scarce/edge
    ds = HuBMAPDataset(
        metadata_df=train_df_subset,
        mode="train",
        load_cached_data=False,
        tissue_overlap_threshold=0.0,
    )

    print(f"Generated {len(ds)} tiles for the first image.")

    if len(ds) > 0:
        # Check item
        sample = ds[0]
        img = sample["image"]
        mask = sample["mask"]

        # Check shapes
        # Image: (3, H, W)
        if img.shape != (3, Config.TILE_SIZE, Config.TILE_SIZE):
            raise AssertionError(f"Image shape mismatch. Got {img.shape}")
        # Mask: (1, H, W)
        if mask.shape != (1, Config.TILE_SIZE, Config.TILE_SIZE):
            raise AssertionError(f"Mask shape mismatch. Got {mask.shape}")

        print("Dataset Item Shapes: Passed")
    else:
        print(
            "Warning: No tiles generated for the sample image (possibly empty tissue mask). Skipping shape check."
        )


def verify_model_and_loss():
    print("\n--- Verifying Model & Loss ---")
    device = torch.device("cpu")  # Use CPU for quick check
    model = FPNResNet18().to(device)
    model.eval()

    # Create dummy input
    B, C, H, W = 2, 3, 256, 256  # Smaller size for speed check
    x = torch.randn(B, C, H, W).to(device)

    # Forward
    with torch.no_grad():
        logits = model(x)

    # Check Output Shape (B, 1, H, W)
    if logits.shape != (B, 1, H, W):
        raise AssertionError(
            f"Model output shape mismatch. Got {logits.shape}, expected {(B, 1, H, W)}"
        )
    print("Model Forward Pass: Passed")

    # Loss
    criterion = BCEDiceLoss()
    y_true = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    loss = criterion(logits, y_true)

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss calculation failed (NaN or negative).")
    print("Loss Calculation: Passed")


def verify_training():
    print("\n--- Verifying Training Loop ---")
    # Adjust Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2

    # Run training in debug mode (uses head(2) of data)
    try:
        run_training(debug=True)
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        # Note: In a real scenario with random weights and 1 epoch, validation dice might be 0.
        # However, best_dice initializes at -1.0, so 0.0 > -1.0, triggering a save.
        raise AssertionError("Model checkpoint was not saved after training.")
    print("Training Loop: Passed")


def verify_inference():
    print("\n--- Verifying Inference ---")

    # Run inference in debug mode (uses head(1) of test data)
    try:
        generate_submission_csv(debug=True)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file not found.")

    df = pd.read_csv(Config.SUBMISSION_PATH)
    if df.empty:
        print("Warning: Submission CSV is empty.")
    else:
        # Check required columns
        if not {"id", "predicted"}.issubset(df.columns):
            raise AssertionError("Submission file missing required columns.")
        print(f"Submission generated with {len(df)} rows.")
    print("Inference: Passed")


if __name__ == "__main__":
    # Setup
    seed_everything(42)

    # 1. Verify Utilities
    verify_utils()

    # 2. Verify Dataset
    # Clean artifacts first to ensure we generate tiles from scratch
    clean_artifacts()
    verify_dataset()

    # 3. Verify Model logic
    verify_model_and_loss()

    # 4. Verify Training
    # Clean artifacts again to ensure training starts fresh (re-generating tiles for the debug subset)
    clean_artifacts()
    verify_training()

    # 5. Verify Inference
    verify_inference()

    print("\nAll verifications passed successfully.")
