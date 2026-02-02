import os
import torch
import numpy as np
import shutil
from library.config import Config
from library.utils import set_seed, rle_encode, GlobalDiceTracker
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.train import train_model

if __name__ == "__main__":
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up demonstration configuration...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Base epochs (train_model debug mode will set this to 2)
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir for demo

    # Ensure working directory is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize environment
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Verify Dataset & Data Loading
    # ==========================================
    print("\n>>> Verifying Dataset...")

    # Initialize dataset
    ds = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        stage="train",
        transform=None,  # No transforms for shape check to be deterministic regarding flips
        cache_dir=os.path.join(Config.WORKING_DIR, "cache_test"),
    )

    # Check length
    assert (
        len(ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(ds)}"

    # Fetch one sample
    img, mask = ds[0]

    # Verify Shapes
    # Image: (C, H, W) -> (6, 256, 256)
    # Mask: (C, H, W) -> (1, 256, 256)
    assert img.shape == (
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Got {img.shape}"
    assert mask.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Mask shape mismatch. Got {mask.shape}"

    # Verify Normalization (approximate bounds [0, 1])
    assert (
        img.min() >= 0.0 and img.max() <= 1.0
    ), f"Image normalization failed. Range: [{img.min()}, {img.max()}]"

    print("Dataset verification passed.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n>>> Verifying Model Architecture...")

    model = ConvNeXtUNet().to(device)
    model.eval()

    # Create dummy input batch: (B, C, H, W)
    dummy_input = torch.randn(
        2, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (B, Num_Classes, H, W)
    expected_shape = (2, Config.NUM_CLASSES, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n>>> Verifying Loss Function...")

    criterion = HybridLoss()

    # Dummy targets: Binary (0 or 1)
    dummy_target = torch.randint(0, 2, expected_shape).float().to(device)

    # Calculate loss
    loss = criterion(output, dummy_target)

    assert loss.ndim == 0, "Loss should be a scalar tensor."
    assert not torch.isnan(loss), "Loss computed as NaN."
    assert loss.item() > 0, "Loss should be positive."

    print(f"Loss computation successful. Value: {loss.item():.4f}")

    # ==========================================
    # 5. Verify Metrics (Global Dice)
    # ==========================================
    print("\n>>> Verifying Global Dice Tracker...")

    tracker = GlobalDiceTracker()

    # Case 1: Perfect Match
    # Logits large positive -> Sigmoid ~ 1.0
    y_pred_perfect = torch.ones(1, 1, 10, 10) * 10.0
    y_true_perfect = torch.ones(1, 1, 10, 10)

    tracker.update(torch.sigmoid(y_pred_perfect), y_true_perfect, threshold=0.5)
    score = tracker.compute()
    assert abs(score - 1.0) < 1e-5, f"Expected Dice 1.0 for perfect match, got {score}"

    # Case 2: Complete Mismatch
    tracker.reset()
    y_pred_wrong = torch.ones(1, 1, 10, 10) * -10.0  # Sigmoid ~ 0.0
    y_true_wrong = torch.ones(1, 1, 10, 10)

    tracker.update(torch.sigmoid(y_pred_wrong), y_true_wrong, threshold=0.5)
    score = tracker.compute()
    assert abs(score - 0.0) < 1e-5, f"Expected Dice 0.0 for mismatch, got {score}"

    print("Metric verification passed.")

    # ==========================================
    # 6. Verify RLE Encoding
    # ==========================================
    print("\n>>> Verifying RLE Encoding...")

    # Create a simple 3x3 mask
    # 0 1 0
    # 0 1 0
    # 0 0 0
    # Column-major flattening:
    # Col 1: 0, 0, 0
    # Col 2: 1, 1, 0
    # Col 3: 0, 0, 0
    # Flattened: 0 0 0 1 1 0 0 0 0
    # Indices (1-based): 4, 5 are 1s.
    # Run: Start at 4, length 2.

    mask_rle = np.zeros((3, 3), dtype=np.uint8)
    mask_rle[0, 1] = 1
    mask_rle[1, 1] = 1

    encoded = rle_encode(mask_rle)
    expected_rle = "4 2"

    assert (
        encoded == expected_rle
    ), f"RLE Encoding incorrect. Expected '{expected_rle}', got '{encoded}'"
    print(f"RLE Encoding verified: {encoded}")

    # ==========================================
    # 7. Run Training Loop (Integration Test)
    # ==========================================
    print("\n>>> Running Training Loop (Debug Mode)...")

    # train_model(debug=True) runs a short loop (2 epochs) on the subset
    # It uses the Config we modified earlier.
    try:
        train_model(debug=True)
        print("Training loop executed successfully.")
    except Exception as e:
        print(f"Training loop failed with error: {e}")
        raise e

    # Verify output file exists
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Best model saved at: {best_model_path}")
    else:
        # It's possible no improvement happened in 2 epochs, but with random init vs pretrained,
        # usually validation loss drops or dice changes.
        # However, if dice stays 0.0 (very hard task on random subset), it might not save "best".
        # We just check if the code ran without crashing.
        print(
            "Training finished (no model saved, likely due to no metric improvement in short run)."
        )

    print("\n>>> Demonstration Complete.")
