import os
import shutil
import torch
import numpy as np
import pandas as pd
import time

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coef, global_dice_score
from library.dataset import ContrailDataset, get_transforms
from library.model import DualStreamUNet
from library.loss import HybridLoss
from library.train import train_model
from library.predict import predict_and_submit


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("[1/7] Configuring environment for rapid execution...")

    # Define separate working directories for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean and recreate directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True  # Enables subsampling of the dataset
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples for training/validation
    Config.NUM_WORKERS = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.BACKBONE = "convnext_tiny"

    # Set random seed for reproducibility
    set_seed(42)
    print("      Configuration updated: Debug Mode=True, Epochs=1, Samples=20")

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("[2/7] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a 2x2 mask: [[0, 1], [0, 0]]
    # Flattened (Fortran/Column-major): [0, 0, 1, 0] -> Indices: 1, 2, 3, 4
    # The '1' is at index 3. Run length is 1. Expected RLE: "3 1"
    mask_simple = np.array([[0, 1], [0, 0]])
    rle_out = rle_encode(mask_simple)
    assert rle_out == "3 1", f"RLE verification failed. Expected '3 1', got '{rle_out}'"

    # Test Dice Coefficient
    # Pred: [1, 1, 0], True: [1, 0, 1]
    # Intersection = 1, Union = 2 + 2 = 4. Dice = 2*1/4 = 0.5
    y_pred = torch.tensor([1.0, 1.0, 0.0])
    y_true = torch.tensor([1.0, 0.0, 1.0])
    dice_val = dice_coef(y_pred, y_true, smooth=0)
    assert (
        abs(dice_val.item() - 0.5) < 1e-5
    ), f"Dice verification failed. Got {dice_val.item()}"

    print("      Utils verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset
    # -------------------------------------------------------------------------
    print("[3/7] Verifying Dataset Loading...")

    # Initialize dataset in debug mode
    ds = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=get_transforms("train"),
        debug=True,
    )

    if len(ds) > 0:
        img, mask, rid = ds[0]

        # Check shapes
        # Image: (Channels, H, W). Channels = 3 (Ash) + 3 (Diff) = 6
        assert img.shape == (
            6,
            256,
            256,
        ), f"Image shape mismatch. Expected (6, 256, 256), got {img.shape}"
        # Mask: (Channels, H, W). Channels = 1
        assert mask.shape == (
            1,
            256,
            256,
        ), f"Mask shape mismatch. Expected (1, 256, 256), got {mask.shape}"
        # Record ID
        assert isinstance(rid, str), "Record ID must be a string."

        print(f"      Dataset loaded sample {rid}. Image shape: {img.shape}")
    else:
        print("      Warning: Dataset is empty. Skipping shape assertions.")

    # -------------------------------------------------------------------------
    # 4. Verify Model
    # -------------------------------------------------------------------------
    print("[4/7] Verifying Model Architecture...")

    model = DualStreamUNet(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_chans_a=Config.IN_CHANNELS_STREAM_A,
        in_chans_b=Config.IN_CHANNELS_STREAM_B,
    )
    model.eval()

    # Create dummy input: Batch=2, Channels=6, H=256, W=256
    dummy_input = torch.randn(2, 6, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: Batch=2, Channels=1 (Binary Class), H=256, W=256
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("      Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Verify Loss
    # -------------------------------------------------------------------------
    print("[5/7] Verifying Loss Function...")

    loss_fn = HybridLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    # Calculate loss
    loss = loss_fn(output, dummy_target)
    assert not torch.isnan(loss), "Loss returned NaN."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"      Loss calculation successful. Value: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Run Training Loop
    # -------------------------------------------------------------------------
    print("[6/7] Executing Training Loop (Demo)...")

    # We call the provided train_model function.
    # It uses the Config we patched earlier (Epochs=1, Debug=True).
    start_time = time.time()
    try:
        train_model(debug=True)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Verify that the model checkpoint was created
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training finished but {Config.BEST_MODEL_PATH} was not created."
        )

    print(f"      Training completed in {time.time() - start_time:.2f}s.")

    # -------------------------------------------------------------------------
    # 7. Run Inference
    # -------------------------------------------------------------------------
    print("[7/7] Executing Inference and Submission Generation...")

    # We call the provided predict_and_submit function.
    # It loads the model from Config.BEST_MODEL_PATH and generates submission.csv.
    try:
        predict_and_submit()
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference finished but {Config.SUBMISSION_PATH} was not created."
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission file created with {len(df_sub)} rows.")

    # Verify submission format
    required_cols = ["record_id", "encoded_pixels"]
    for col in required_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    print("      Inference completed successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
