import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library modules
from library import config
from library import data_utils
from library import dataset
from library import model
from library import loss
from library import train_engine
from library import inference_engine


def main():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Patching Configuration for Fast Execution...")

    # Set fixed seed
    train_engine.set_seed(config.SEED)

    # Modify Training Params: 1 epoch, debug mode (tiny subset), small batch
    config.TRAINING_PARAMS["epochs"] = 1
    config.TRAINING_PARAMS["debug"] = True
    config.TRAINING_PARAMS["batch_size"] = 2
    config.TRAINING_PARAMS["valid_threshold"] = 0.0  # Force save even if score is low

    # Modify Specialist Settings: Only run 'Mid' specialist to save time
    # We keep only the 'Mid' key in the dictionary
    mid_settings = config.SPECIALIST_SETTINGS["Mid"]
    config.SPECIALIST_SETTINGS = {"Mid": mid_settings}

    print("Configuration patched: Running 'Mid' specialist for 1 epoch (debug mode).")

    # -------------------------------------------------------------------------
    # 2. Data Utils Verification (Slab Generation)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Utils (Slab Generation)...")

    # We use fragment '1' from training set for demonstration
    frag_id = "1"
    z_start = mid_settings["z_start"]
    z_end = mid_settings["z_end"]

    # Generate/Load slab
    slab = data_utils.get_fragment_3ch_slab(
        fragment_id=frag_id,
        split="train",
        z_start=z_start,
        z_end=z_end,
        slab_params=config.SLAB_PARAMS,
        load_cached_data=True,
    )

    # Assertions
    assert isinstance(slab, np.ndarray), "Slab must be a numpy array"
    assert slab.ndim == 3, f"Slab must be 3D (H, W, C), got {slab.ndim}"
    assert slab.shape[2] == 3, f"Slab must have 3 channels, got {slab.shape[2]}"
    assert slab.dtype == np.float32, f"Slab must be float32, got {slab.dtype}"
    assert (
        slab.min() >= 0.0 and slab.max() <= 1.0
    ), "Slab values must be normalized [0, 1]"

    print(
        f"Slab generation successful. Shape: {slab.shape}, Range: [{slab.min():.2f}, {slab.max():.2f}]"
    )

    # -------------------------------------------------------------------------
    # 3. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying InkDataset...")

    # Load metadata
    df_train = pd.read_csv(config.PATHS.TRAIN_METADATA)
    # Take a tiny subset for manual verification
    df_subset = df_train.head(4)

    ds = dataset.InkDataset(metadata=df_subset, specialist_mode="Mid", split="train")

    # Fetch one sample
    image_t, label_t = ds[0]

    # Assertions
    assert torch.is_tensor(image_t), "Image must be a tensor"
    assert torch.is_tensor(label_t), "Label must be a tensor"
    # Shape check: (C, H, W)
    assert image_t.shape == (
        3,
        config.TILE_SIZE,
        config.TILE_SIZE,
    ), f"Unexpected image shape: {image_t.shape}"
    assert label_t.shape == (
        1,
        config.TILE_SIZE,
        config.TILE_SIZE,
    ), f"Unexpected label shape: {label_t.shape}"

    print("Dataset verification successful. Sample shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model and Loss...")

    # Instantiate Model
    net = model.get_model(config.MODEL_PARAMS)
    net.to(config.DEVICE)

    # Create dummy batch (B, C, H, W)
    dummy_input = image_t.unsqueeze(0).to(config.DEVICE)  # Batch size 1
    dummy_target = label_t.unsqueeze(0).to(config.DEVICE)

    # Forward Pass
    logits = net(dummy_input)
    assert (
        logits.shape == dummy_target.shape
    ), f"Logits shape {logits.shape} mismatch with target {dummy_target.shape}"

    # Loss Calculation
    criterion = loss.BCEDiceLoss()
    loss_val = criterion(logits, dummy_target)

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() >= 0, "Loss should be non-negative"

    print(
        f"Model forward pass and Loss calculation successful. Loss: {loss_val.item():.4f}"
    )

    # Clean up memory
    del net, dummy_input, dummy_target, logits, loss_val
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Specialist: Mid)...")

    # This runs the full training logic for 1 epoch on debug data
    train_engine.run_specialist_training("Mid", load_cached_data=True)

    # Verify checkpoint creation
    expected_ckpt = os.path.join(config.PATHS.WORKING_DIR, "model_Mid.pth")
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"

    print("Training loop completed. Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Inference Execution
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference Engine...")

    # Initialize Engine
    engine = inference_engine.InferenceEngine()

    # Run Generation
    # This loads the 'Mid' model we just trained and predicts on the test set
    engine.generate_submission()

    # Verify Submission File
    submission_path = config.PATHS.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Inference completed. Submission generated with {len(df_sub)} rows.")
    print("Sample row:")
    print(df_sub.head(1))

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
