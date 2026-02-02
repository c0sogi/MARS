import os
import sys
import numpy as np
import torch
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode
from library.stain_deconv import StainDeconvolution
from library.dataset import HuBMAPDataset
from library.model import StainNet
from library.loss import DeepSupervisionLoss
from library.train import train_model
from library.inference import predict_test_set

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("=== Starting HuBMAP Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("[1] Configuring environment...")
    # Patch Config to run a fast, minimal version
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small subset of tiles
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Setup directories
    Config.setup()
    print(f"    Working directory set to: {Config.WORKING_DIR}")
    print("    Debug mode enabled. Epochs: 1, Batch Size: 2")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utils (RLE Encoding/Decoding)...")
    # Create a synthetic 10x10 binary mask with a 2x2 square
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[2:4, 2:4] = 1  # Pixels at (2,2), (2,3), (3,2), (3,3)

    # Encode
    rle_str = rle_encode(dummy_mask)
    # Decode
    decoded_mask = rle_decode(rle_str, (10, 10))

    # Verify
    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError(
            "RLE Round-trip failed: Decoded mask does not match original."
        )
    print("    RLE encoding/decoding verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Stain Deconvolution
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Stain Deconvolution Layer...")
    stain_layer = StainDeconvolution()
    # Create random RGB input: (Batch=2, Channels=3, H=256, W=256)
    dummy_rgb = torch.rand(2, 3, 256, 256)

    with torch.no_grad():
        output = stain_layer(dummy_rgb)

    # Expected output: (Batch=2, Channels=5, H=256, W=256)
    # Channels are R, G, B, Hematoxylin, Eosin
    if output.shape != (2, 5, 256, 256):
        raise AssertionError(
            f"Stain Deconvolution output shape mismatch. Expected (2, 5, 256, 256), got {output.shape}"
        )
    print("    Stain Deconvolution shape check passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset Loading...")
    # Initialize dataset in train mode (uses metadata/train.csv)
    # Note: This will generate/load tile cache in ./working/demo_run/cache
    dataset = HuBMAPDataset(mode="train", load_cached_data=False)

    if len(dataset) == 0:
        raise AssertionError("Dataset is empty. Check metadata or debug sample size.")

    print(f"    Dataset initialized with {len(dataset)} tiles (Debug Mode).")

    # Fetch one sample
    img, mask = dataset[0]

    # Check types and shapes
    # Image should be tensor (3, H, W) due to Albumentations ToTensorV2
    # Mask should be tensor (1, H, W)
    if not isinstance(img, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Dataset items must be torch Tensors.")

    if img.shape[0] != 3:
        raise AssertionError(
            f"Image tensor has wrong channel count: {img.shape[0]} (Expected 3)"
        )

    if mask.shape[0] != 1:
        raise AssertionError(
            f"Mask tensor has wrong channel count: {mask.shape[0]} (Expected 1)"
        )

    print(f"    Sample retrieved. Image: {img.shape}, Mask: {mask.shape}")

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture (StainNet)...")
    model = StainNet()
    model.eval()

    # Create a dummy batch matching dataset output
    dummy_batch = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE)

    with torch.no_grad():
        preds = model(dummy_batch)

    # Check Deep Supervision output
    if Config.DEEP_SUPERVISION:
        if not isinstance(preds, list):
            raise AssertionError(
                "Model should return a list when Deep Supervision is enabled."
            )
        main_pred = preds[0]
    else:
        main_pred = preds

    expected_shape = (2, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    if main_pred.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {main_pred.shape}"
        )

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 6. Verify Loss Function
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Loss Function (DeepSupervisionLoss)...")
    loss_fn = DeepSupervisionLoss()

    # Create dummy targets
    dummy_targets = torch.randint(
        0, 2, (2, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    ).float()

    # Calculate loss
    loss = loss_fn(preds, dummy_targets)

    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        raise AssertionError("Loss must be a scalar tensor.")

    if torch.isnan(loss):
        raise AssertionError("Loss is NaN.")

    print(f"    Loss calculation successful. Value: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 7. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[7] Running Training Loop (1 Epoch)...")
    # train_model() uses the Config class we patched
    try:
        train_model(load_cached_data=False)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        # Note: In the provided train.py, model is only saved if validation improves.
        # Since we run 1 epoch with random init, it might not save if val dice is 0.
        # However, usually val dice > 0 or best_dice starts at 0.0.
        # If val dice is 0, it won't save.
        # Let's check if the file exists, if not, we warn but don't fail the demo
        # as it depends on random convergence.
        print("    Notice: Model file was not saved (Validation Dice did not improve).")
        # For the sake of the next step (Inference), we need a model file.
        # We will manually save the model state here to ensure inference can run.
        print("    Manually saving model for inference demonstration...")
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        print("    Model saved successfully.")

    # -------------------------------------------------------------------------
    # 8. Run Inference Loop
    # -------------------------------------------------------------------------
    print("\n[8] Running Inference on Test Set...")
    try:
        predict_test_set(load_cached_data=False)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    if "id" not in df_sub.columns or "predicted" not in df_sub.columns:
        raise AssertionError(
            "Submission file missing required columns ('id', 'predicted')."
        )

    print(f"    Submission generated at {Config.SUBMISSION_PATH}")
    print(f"    Rows: {len(df_sub)}")
    print("    Sample:")
    print(df_sub.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    set_seed(42)
    run_demo()
