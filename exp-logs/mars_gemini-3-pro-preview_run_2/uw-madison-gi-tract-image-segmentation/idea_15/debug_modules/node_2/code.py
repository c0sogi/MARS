import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, dice_coef, hausdorff_distance_3d
from library.dataset import get_processed_dataframe, UWMapDataset
from library.model import UnetPlusPlus
from library.loss import BCETverskyLoss
from library.train import train_model
from library.inference import inference_fn


def main():
    print("=== Starting Demo Script ===")

    # 1. Setup Configuration for Demo
    # We override the default configuration to run a fast, lightweight demo.
    print("\n[1] Configuring environment...")

    # Create a specific directory for this demo run
    demo_dir = os.path.join("working", "demo_execution")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Patch the Config class
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size for CPU/memory safety
    Config.DEBUG = True  # Enable debug mode in internal functions
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.SEED)
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Verify Metrics
    print("\n[2] Verifying Metrics...")

    # Case A: Identical masks
    mask_a = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_a[2:5, 2:5, 2:5] = 1
    mask_b = mask_a.copy()

    dice = dice_coef(mask_a, mask_b)
    hd = hausdorff_distance_3d(mask_a, mask_b)

    assert np.isclose(dice, 1.0), f"Dice should be 1.0 for identical masks, got {dice}"
    assert np.isclose(hd, 0.0), f"Hausdorff should be 0.0 for identical masks, got {hd}"

    # Case B: Disjoint masks
    mask_c = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_c[7:9, 7:9, 7:9] = 1

    dice_disjoint = dice_coef(mask_a, mask_c)
    assert np.isclose(
        dice_disjoint, 0.0
    ), f"Dice should be 0.0 for disjoint masks, got {dice_disjoint}"

    print("Metrics logic verified.")

    # 3. Verify Dataset Loading
    print("\n[3] Verifying Dataset pipeline...")

    # Load metadata dataframe
    df_train = get_processed_dataframe(Config.TRAIN_METADATA_PATH, split_name="train")
    assert len(df_train) > 0, "Train dataframe is empty."

    # Create a small dataset instance
    # We take a small slice to avoid loading too many images
    ds_subset = UWMapDataset(df_train.iloc[:10], mode="train", img_size=256)

    # Fetch one sample
    img_tensor, mask_tensor = ds_subset[0]

    # Verify shapes
    # Image: (3, H, W) -> 3 channels because of 2.5D stacking (prev, curr, next slices)
    # Mask: (3, H, W) -> 3 channels for classes (large_bowel, small_bowel, stomach)
    print(f"Sample Image Shape: {img_tensor.shape}")
    print(f"Sample Mask Shape: {mask_tensor.shape}")

    assert img_tensor.shape == (
        3,
        256,
        256,
    ), f"Expected image (3, 256, 256), got {img_tensor.shape}"
    assert mask_tensor.shape == (
        3,
        256,
        256,
    ), f"Expected mask (3, 256, 256), got {mask_tensor.shape}"
    assert img_tensor.dtype == torch.float32, "Image tensor should be float32"

    print("Dataset pipeline verified.")

    # 4. Verify Model and Loss
    print("\n[4] Verifying Model and Loss...")

    device = torch.device(Config.DEVICE)
    model = UnetPlusPlus().to(device)
    criterion = BCETverskyLoss().to(device)

    # Create dummy batch (B, C, H, W)
    dummy_img = torch.randn(2, 3, 256, 256).to(device)
    dummy_mask = torch.randint(0, 2, (2, 3, 256, 256)).float().to(device)

    # Forward pass
    model.train()  # Enable Deep Supervision
    outputs = model(dummy_img)

    # With deep supervision, output should be a list of tensors
    assert isinstance(
        outputs, list
    ), "Model in train mode should return a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs from Deep Supervision, got {len(outputs)}"
    assert outputs[-1].shape == (
        2,
        3,
        256,
        256,
    ), f"Final output shape mismatch: {outputs[-1].shape}"

    # Loss calculation
    loss = criterion(outputs, dummy_mask)
    print(f"Calculated Dummy Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verified.")

    # 5. Run Training Loop (Debug Mode)
    print("\n[5] Running Training Loop (Debug Mode)...")
    # This calls the library function which handles the loop, validation, and saving.
    # debug=True ensures it uses a tiny subset and runs for limited epochs.
    train_model(debug=True)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Training did not produce 'best_model.pth'"
    print("Training loop completed successfully.")

    # 6. Run Inference Loop (Debug Mode)
    print("\n[6] Running Inference Loop (Debug Mode)...")
    # This generates the submission file using the model trained in step 5.
    inference_fn(debug=True)

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Inference did not produce submission file"

    # Validate submission format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission file generated with {len(sub_df)} rows.")

    expected_cols = ["id", "class", "predicted"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check content of first row
    first_row = sub_df.iloc[0]
    print(f"Sample Submission Row: {first_row.to_dict()}")

    print("Inference loop completed successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
