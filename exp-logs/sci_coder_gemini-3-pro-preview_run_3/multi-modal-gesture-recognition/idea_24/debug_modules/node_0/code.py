import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import json

# Import the provided library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import trainer


def run_demo():
    print("=== Starting Demonstration of Gesture Recognition Pipeline ===")

    # ==========================================
    # 1. Configure for Fast Demonstration
    # ==========================================
    print("\n[Step 1] Configuring environment for demo...")

    # Override config parameters for speed and isolation
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = os.path.join(demo_dir, "cache")
    config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Set debug mode to process only a few samples
    config.DEBUG = True
    config.SUBSET_SIZE = 10
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4

    # Ensure reproducibility
    trainer.set_seed(config.SEED)

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Subset Size: {config.SUBSET_SIZE}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[Step 2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Sequence: Background(0) -> Gesture 1 -> Gesture 1 -> Background(0) -> Gesture 2 -> Gesture 2
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0]
    expected_rle = [1, 2]
    rle_result = utils.rle_encode(raw_preds)
    print(f"RLE Input: {raw_preds}")
    print(f"RLE Output: {rle_result}")
    assert (
        rle_result == expected_rle
    ), f"RLE failed. Expected {expected_rle}, got {rle_result}"

    # Test Truncated MSE Loss
    # Create two identical log_prob tensors -> Loss should be 0
    t1 = torch.zeros((1, 5, 21))
    loss_val = utils.truncated_mse_loss(t1)
    assert loss_val.item() == 0.0, "Truncated MSE should be 0 for constant input"
    print("Utility functions verified.")

    # ==========================================
    # 3. Verify Dataset Loading & Processing
    # ==========================================
    print("\n[Step 3] Verifying Dataset...")

    # Initialize Dataset (Train mode)
    # This will trigger load_raw_data, which uses config.SUBSET_SIZE
    ds_train = dataset.GestureDataset(mode="train", load_cached_data=False)

    print(f"Dataset initialized. Number of windows: {len(ds_train)}")

    if len(ds_train) > 0:
        # Fetch one sample
        features, labels = ds_train[0]

        # Verify shapes
        # Features: (WINDOW_SIZE, INPUT_DIM) -> (64, 193)
        # Labels: (WINDOW_SIZE,) -> (64,)
        print(f"Sample Feature Shape: {features.shape}")
        print(f"Sample Label Shape: {labels.shape}")

        expected_feat_shape = (config.WINDOW_SIZE, config.INPUT_DIM)
        expected_label_shape = (config.WINDOW_SIZE,)

        assert (
            features.shape == expected_feat_shape
        ), f"Feature shape mismatch. Expected {expected_feat_shape}, got {features.shape}"
        assert (
            labels.shape == expected_label_shape
        ), f"Label shape mismatch. Expected {expected_label_shape}, got {labels.shape}"

        # Verify Data Type
        assert features.dtype == torch.float32, "Features should be float32"
        assert labels.dtype == torch.int64, "Labels should be int64"

        # Verify Augmentation (Train mode)
        # Fetching the same index again should result in slightly different features due to rotation/scaling
        features_aug, _ = ds_train[0]
        if not torch.allclose(features, features_aug):
            print("Augmentation verified: Features vary between calls.")
        else:
            print("Note: Features identical. Augmentation might be subtle or disabled.")

    else:
        print("Warning: Dataset is empty. Check input data availability.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[Step 4] Verifying Model Architecture...")

    net = model.GI_HCSN()
    net.eval()  # Set to eval to disable dropout for deterministic shape check

    # Create dummy input batch: (Batch, Time, InputDim)
    dummy_input = torch.randn(2, config.WINDOW_SIZE, config.INPUT_DIM)

    # Forward pass
    logits_1, logits_2, logits_3 = net(dummy_input)

    print("Forward pass successful.")
    print(f"Stage 1 Output Shape: {logits_1.shape}")
    print(f"Stage 2 Output Shape: {logits_2.shape}")
    print(f"Stage 3 Output Shape: {logits_3.shape}")

    expected_out_shape = (2, config.WINDOW_SIZE, config.NUM_CLASSES)

    assert logits_1.shape == expected_out_shape, "Stage 1 output shape mismatch"
    assert logits_2.shape == expected_out_shape, "Stage 2 output shape mismatch"
    assert logits_3.shape == expected_out_shape, "Stage 3 output shape mismatch"

    # ==========================================
    # 5. Verify Training Loop
    # ==========================================
    print("\n[Step 5] Demonstrating Training Loop...")

    # Initialize Trainer
    # Trainer internally initializes datasets again, but will use the cache generated in Step 3
    # or regenerate quickly since DEBUG=True.
    demo_trainer = trainer.Trainer()

    # Run training
    print("Starting fit()...")
    demo_trainer.fit()

    # Verify model checkpoint creation
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"Success: Model checkpoint saved at {config.BEST_MODEL_PATH}")
        file_size = os.path.getsize(config.BEST_MODEL_PATH)
        print(f"Checkpoint size: {file_size / 1024:.2f} KB")
    else:
        raise AssertionError("Model checkpoint was not created!")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n[Step 6] Demonstrating Inference...")

    # Load the best model
    loaded_model = model.GI_HCSN()
    loaded_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    loaded_model.eval()

    # Run inference on a sample from the dataset
    if len(ds_train) > 0:
        sample_feats, sample_labels = ds_train[0]
        # Add batch dimension
        input_tensor = sample_feats.unsqueeze(0)  # (1, T, D)

        with torch.no_grad():
            _, _, logits_final = loaded_model(input_tensor)

        # Get predictions
        preds = torch.argmax(logits_final, dim=2).squeeze(0).numpy()

        # Decode
        predicted_gestures = utils.rle_encode(preds)
        true_gestures = utils.rle_encode(sample_labels.numpy())

        print(f"Sample True Gestures (IDs): {true_gestures}")
        print(f"Sample Predicted Gestures (IDs): {predicted_gestures}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
