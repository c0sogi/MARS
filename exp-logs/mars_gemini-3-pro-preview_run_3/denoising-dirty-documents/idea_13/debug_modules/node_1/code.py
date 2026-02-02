import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_rmse, apply_tta, reverse_tta
from library.model import EZ_ResDnCNN
from library.dataset import get_dataloaders
from library.train_engine import run_training
from library.inference_engine import (
    load_ensemble,
    predict_image,
    create_submission_file,
)


def run_demo():
    print("Starting Demo Execution...")

    # 1. Setup Environment and Config Overrides for Speed
    # We override the Config class attributes directly to create a "mini" environment
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Configuring demo environment in {demo_dir}...")

    # Override paths
    Config.WORKING_DIR = demo_dir
    Config.CACHE_TRAIN_PATCHES = os.path.join(demo_dir, "train_patches.npy")
    Config.CACHE_TRAIN_TARGETS = os.path.join(demo_dir, "train_targets.npy")
    Config.CACHE_VAL_PATCHES = os.path.join(demo_dir, "val_patches.npy")
    Config.CACHE_VAL_TARGETS = os.path.join(demo_dir, "val_targets.npy")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Override Model/Training Hyperparameters for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_RES_BLOCKS = 2  # Shallow network for fast demo
    Config.PATCH_SIZE = 32  # Smaller patches
    Config.STRIDE = 100  # Large stride to get very few patches per image
    Config.ENSEMBLE_SIZE = 1
    Config.TTA_STEPS = 2  # Reduce TTA steps
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo

    # 2. Create Mini Metadata
    # We read the original metadata and create subsets to ensure the demo runs instantly
    print("Creating mini metadata files...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Subset
    df_train = orig_train.head(5)  # Only 5 training images
    df_val = orig_val.head(2)  # Only 2 validation images
    df_test = orig_test.head(2)  # Only 2 test images

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    df_train.to_csv(mini_train_path, index=False)
    df_val.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    # Point Config to mini metadata
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # 3. Verify Utils
    print("Verifying Utils...")
    seed_everything(42)

    # RMSE Check
    y_true = np.array([1.0, 0.5, 0.0])
    y_pred = np.array([1.0, 0.5, 0.0])
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 0.0, f"RMSE should be 0, got {rmse}"

    y_pred_off = np.array([0.0, 1.5, 1.0])  # diffs: 1, 1, 1 -> mse 1 -> rmse 1
    rmse_off = calculate_rmse(y_true, y_pred_off)
    assert np.isclose(rmse_off, 1.0), f"RMSE should be 1.0, got {rmse_off}"

    # TTA Check
    # Create a 1x1x2x2 tensor: [[1, 2], [3, 4]]
    t = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    # k=4 is Horizontal Flip -> [[2, 1], [4, 3]]
    t_aug = apply_tta(t, k=4)
    assert t_aug[0, 0, 0, 0] == 2.0
    assert t_aug[0, 0, 0, 1] == 1.0
    # Reverse
    t_rev = reverse_tta(t_aug, k=4)
    assert torch.allclose(t, t_rev), "Reverse TTA failed to restore original tensor"

    # 4. Verify Dataset Loading
    print("Verifying Dataset Loading...")
    # This will trigger patch extraction and caching
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    assert len(train_loader) > 0, "Train loader is empty"

    # Check batch structure
    batch_inputs, batch_targets = next(iter(train_loader))
    print(f"Batch Shape: {batch_inputs.shape}")
    assert batch_inputs.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    assert batch_targets.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )

    # 5. Verify Model Architecture
    print("Verifying Model...")
    device = torch.device(Config.DEVICE)
    model = EZ_ResDnCNN().to(device)

    # Forward pass check
    dummy_input = batch_inputs.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"

    # 6. Verify Training Engine
    print("Verifying Training Engine...")
    # We name it "model_0" so inference engine can find it later (it looks for model_{i}.pth)
    model_name = "model_0"
    best_rmse = run_training(
        model_name, epochs=Config.NUM_EPOCHS, load_cached_data=True, debug=True
    )

    saved_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}.pth")
    assert os.path.exists(
        saved_model_path
    ), f"Model checkpoint not found at {saved_model_path}"
    assert isinstance(best_rmse, float)

    # 7. Verify Inference Engine
    print("Verifying Inference Engine...")

    # Load Ensemble
    models = load_ensemble(device, ensemble_size=1)
    assert len(models) == 1, "Failed to load the trained model."

    # Predict Single Image
    test_img_path = os.path.join(Config.INPUT_DIR, df_test.iloc[0]["input_path"])
    denoised_img = predict_image(
        test_img_path, models, device, tta_steps=Config.TTA_STEPS
    )

    # Check output properties
    assert isinstance(denoised_img, np.ndarray)
    assert denoised_img.ndim == 2, "Prediction should be 2D (grayscale)"
    assert (
        denoised_img.min() >= 0.0 and denoised_img.max() <= 1.0
    ), "Prediction values out of range [0, 1]"

    # Create Submission
    print("Generating Submission...")
    create_submission_file()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in sub_df.columns and "value" in sub_df.columns
    assert len(sub_df) > 0

    print("\nDemo Execution Completed Successfully!")
    print(f"Artifacts stored in: {Config.WORKING_DIR}")


if __name__ == "__main__":
    run_demo()
