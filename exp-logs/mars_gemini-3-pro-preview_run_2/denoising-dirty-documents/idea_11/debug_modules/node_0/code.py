import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil


# 1. Suppress tqdm progress bars before importing library modules that use it
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import library.inference

library.inference.tqdm = silent_tqdm

# Import provided library modules
from library.config import Config
from library.utils import set_seed, count_parameters
from library.dataset import DenoisingDataset
from library.layers import CoordinateAttention, SKFusion, CSKBlock, ASPP
from library.model import CSKResUNet
import library.train as train_module
import library.inference as inference_module


def run_demo():
    print("=== Starting Demo Execution ===")

    # --- Configuration Override for Demo ---
    # We modify the Config class attributes directly to create a fast execution environment
    print("Configuring demo parameters...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_test.csv")

    # Create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.PATCHES_PER_IMAGE = 2  # Reduced from 100
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 4  # Only use 4 images
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo
    Config.USE_TTA = False  # Disable TTA for speed

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --- 1. Verify Dataset Logic ---
    print("\n--- Verifying Dataset Logic ---")

    # Test Train Dataset (Patch-based)
    ds_train = DenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="train",
        load_cached_data=False,  # Force processing
        limit_size=Config.DEBUG_SUBSET_SIZE,
    )

    expected_len = Config.DEBUG_SUBSET_SIZE * Config.PATCHES_PER_IMAGE
    print(f"Train Dataset Length: {len(ds_train)} (Expected: {expected_len})")
    assert (
        len(ds_train) == expected_len
    ), "Train dataset length calculation is incorrect."

    # Fetch a sample
    noisy_patch, clean_patch, img_id = ds_train[0]
    print(
        f"Train Sample Shapes - Noisy: {noisy_patch.shape}, Clean: {clean_patch.shape}"
    )

    assert noisy_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect noisy patch shape."
    assert clean_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect clean patch shape."
    assert isinstance(noisy_patch, torch.Tensor), "Output should be a tensor."

    # Test Val Dataset (Full Image)
    ds_val = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="val",
        load_cached_data=False,
        limit_size=2,
    )
    noisy_img, clean_img, _ = ds_val[0]
    print(f"Val Sample Shapes - Noisy: {noisy_img.shape}, Clean: {clean_img.shape}")
    # Validation images are full size, just check channel dim is 1 and it's 3D tensor
    assert (
        noisy_img.dim() == 3 and noisy_img.size(0) == 1
    ), "Incorrect val image tensor format."

    # --- 2. Verify Layers Logic ---
    print("\n--- Verifying Custom Layers ---")
    device = torch.device("cpu")
    dummy_input = torch.randn(2, 64, 32, 32).to(device)  # B, C, H, W

    # Coordinate Attention
    ca = CoordinateAttention(in_channels=64, reduction=16).to(device)
    out_ca = ca(dummy_input)
    assert (
        out_ca.shape == dummy_input.shape
    ), f"CoordinateAttention output shape mismatch: {out_ca.shape}"
    print("CoordinateAttention: OK")

    # SK Fusion
    sk = SKFusion(in_channels=64, out_channels=64).to(device)
    out_sk = sk(dummy_input)
    assert (
        out_sk.shape == dummy_input.shape
    ), f"SKFusion output shape mismatch: {out_sk.shape}"
    print("SKFusion: OK")

    # CSK Block
    csk = CSKBlock(in_channels=64, out_channels=128, stride=2).to(device)
    out_csk = csk(dummy_input)
    # Stride 2 reduces 32x32 -> 16x16, channels 64 -> 128
    assert out_csk.shape == (
        2,
        128,
        16,
        16,
    ), f"CSKBlock output shape mismatch: {out_csk.shape}"
    print("CSKBlock (stride 2): OK")

    # ASPP
    aspp = ASPP(in_channels=64, out_channels=64).to(device)
    out_aspp = aspp(dummy_input)
    assert out_aspp.shape == (
        2,
        64,
        32,
        32,
    ), f"ASPP output shape mismatch: {out_aspp.shape}"
    print("ASPP: OK")

    # --- 3. Verify Model Logic ---
    print("\n--- Verifying Full Model ---")
    model = CSKResUNet().to(device)
    # Input to model is 1 channel (grayscale)
    dummy_model_input = torch.randn(2, 1, 128, 128).to(device)

    with torch.no_grad():
        out_model = model(dummy_model_input)

    print(f"Model Output Shape: {out_model.shape}")
    assert out_model.shape == (2, 1, 128, 128), "Model output shape mismatch."

    param_count = count_parameters(model)
    print(f"Model Parameters: {param_count}")
    assert param_count > 0, "Model has no parameters."

    # --- 4. Execute Training Loop ---
    print("\n--- Executing Training Loop (Demo) ---")
    # run_training uses Config settings we modified earlier
    train_module.run_training()

    # Verify checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print(f"Checkpoint verified at: {Config.MODEL_SAVE_PATH}")

    # --- 5. Execute Inference ---
    print("\n--- Executing Inference (Demo) ---")
    # run_inference uses Config settings and loads the model we just trained
    inference_module.run_inference(limit_size=2)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df.head()}")

    assert list(df.columns) == ["id", "value"], "Submission columns are incorrect."
    assert len(df) > 0, "Submission file is empty."
    assert (
        df["value"].dtype == float or df["value"].dtype == int
    ), "Value column has incorrect type."

    # Check ID format (e.g., "110_1_1")
    sample_id = df.iloc[0]["id"]
    assert len(sample_id.split("_")) == 3, f"ID format incorrect: {sample_id}"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
