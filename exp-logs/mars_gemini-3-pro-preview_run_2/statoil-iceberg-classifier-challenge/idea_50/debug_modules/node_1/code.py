import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import from the provided library
import library.config as config
import library.utils as utils
import library.layers as layers
import library.model as model_lib
import library.data_loader as data_loader
import library.trainer as trainer


def run_demo():
    print("=== Starting Library Demo ===\n")

    # 1. Setup and Reproducibility
    print("--- 1. Setting Random Seeds ---")
    utils.set_seed(config.SEED)
    print("Seeds set successfully.\n")

    # 2. Verify Utility Functions
    print("--- 2. Verifying Utility Functions (Global Stats) ---")
    # This function calculates or loads cached stats for image normalization
    # It reads the full train.json if cache is missing, which might take a few seconds.
    stats = utils.calculate_global_stats(load_cached_data=True)

    print("Global Stats computed:")
    print(stats)

    expected_keys = ["b1_min", "b1_max", "b2_min", "b2_max", "b3_min", "b3_max"]
    for k in expected_keys:
        assert k in stats, f"Missing key {k} in global stats"
        assert isinstance(stats[k], float), f"Stat {k} is not a float"

    # Logic check: Max should be greater than Min
    assert stats["b1_max"] > stats["b1_min"], "Band 1 max is not greater than min"
    assert stats["b2_max"] > stats["b2_min"], "Band 2 max is not greater than min"
    print("Utility functions verified.\n")

    # 3. Verify Custom Layers
    print("--- 3. Verifying Custom Layers ---")

    # 3a. DualPooling
    # Input: (Batch, Channels, Height, Width)
    # Logic: MaxPool and MinPool (via negated MaxPool) concatenated -> Channels * 2
    batch_size = 2
    channels = 4
    size = 10
    x_input = torch.randn(batch_size, channels, size, size)

    dual_pool = layers.DualPooling(kernel_size=2, stride=2)
    x_out = dual_pool(x_input)

    expected_shape = (batch_size, channels * 2, size // 2, size // 2)
    print(f"DualPooling Input: {x_input.shape}, Output: {x_out.shape}")
    assert (
        x_out.shape == expected_shape
    ), f"DualPooling output shape mismatch. Expected {expected_shape}, got {x_out.shape}"

    # 3b. CBAM (Attention Module)
    # Logic: Should maintain input shape
    cbam = layers.CBAM(planes=channels, ratio=2)
    x_out_cbam = cbam(x_input)
    print(f"CBAM Input: {x_input.shape}, Output: {x_out_cbam.shape}")
    assert x_out_cbam.shape == x_input.shape, "CBAM should preserve input shape"

    # 3c. WideConvBlock
    # Logic: Conv 3x3 -> BN -> ReLU
    wide_block = layers.WideConvBlock(in_channels=channels, out_channels=8)
    x_out_block = wide_block(x_input)
    print(f"WideConvBlock Input: {x_input.shape}, Output: {x_out_block.shape}")
    assert x_out_block.shape == (
        batch_size,
        8,
        size,
        size,
    ), "WideConvBlock output shape mismatch"

    print("Custom layers verified.\n")

    # 4. Verify Model Architecture
    print("--- 4. Verifying Model Architecture (CA_WBN) ---")
    net = model_lib.CA_WBN()

    # Dummy inputs matching the dataset structure
    # Image: (Batch, 3, 75, 75)
    # Meta: (Batch, 1) or (Batch,)
    dummy_img = torch.randn(4, 3, 75, 75)
    dummy_meta = torch.randn(4, 1)

    # Forward pass
    output = net(dummy_img, dummy_meta)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(
        output <= 1
    ), "Model output (sigmoid) not in [0, 1]"

    print("Model architecture verified.\n")

    # 5. Verify Data Loading
    print("--- 5. Verifying Data Loaders ---")
    # Use debug=True to load a small subset of data
    train_loader, val_loader = data_loader.get_loaders(fold=0, debug=True)

    print(f"Train Loader Batch Size: {train_loader.batch_size}")
    print(f"Train Loader Length (batches): {len(train_loader)}")

    # Fetch one batch
    imgs, metas, targets, ids = next(iter(train_loader))

    print(f"Batch Images Shape: {imgs.shape}")
    print(f"Batch Metas Shape: {metas.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Assertions
    assert (
        imgs.dim() == 4 and imgs.shape[1] == 3
    ), "Image tensor has incorrect dimensions"
    assert (
        imgs.shape[2] == 75 and imgs.shape[3] == 75
    ), "Image tensor has incorrect spatial resolution"
    assert targets.shape[0] == imgs.shape[0], "Target batch size mismatch"
    assert isinstance(ids, tuple) or isinstance(ids, list), "IDs should be a list/tuple"

    # Verify Test Loader
    test_loader = data_loader.get_test_loader(debug=True)
    test_imgs, test_metas, test_ids = next(iter(test_loader))
    assert test_imgs.shape[1] == 3, "Test images incorrect channel count"

    print("Data loaders verified.\n")

    # 6. Verify Training Loop (Simulation)
    print("--- 6. Verifying Training Loop ---")

    # Monkey-patch configuration for speed
    # We want to run 1 epoch on 1 fold only for demonstration
    trainer.MAX_EPOCHS = 1
    trainer.NUM_FOLDS = 1

    # Run training
    # debug=True ensures we use the small subset loaded in step 5 logic
    print("Starting training run (Debug Mode: 1 Epoch, 1 Fold)...")
    trainer.run_training(debug=True)

    # Verify model artifact creation
    expected_model_path = os.path.join(config.MODEL_DIR, "model_fold_0.pth")
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"Training completed. Model saved to {expected_model_path}\n")

    # 7. Verify Inference
    print("--- 7. Verifying Inference ---")

    # Load the trained model
    model = model_lib.CA_WBN().to(config.DEVICE)
    model.load_state_dict(torch.load(expected_model_path, map_location=config.DEVICE))
    model.eval()

    # Run inference on the test batch fetched earlier
    with torch.no_grad():
        test_imgs = test_imgs.to(config.DEVICE)
        test_metas = test_metas.to(config.DEVICE)
        preds = model(test_imgs, test_metas)

    print(f"Prediction Shape: {preds.shape}")
    print(f"Sample Predictions: {preds.flatten()[:5].cpu().numpy()}")

    assert preds.shape[0] == test_imgs.shape[0], "Prediction batch size mismatch"

    print("Inference verified.\n")
    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
