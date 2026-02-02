import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import glob

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_weighted_log_loss, get_device
from library.dicom_preprocessor import preprocess_and_cache
from library.dataset import CervicalSpineDataset
from library.model import CervicalMILModel
from library.loss import ImplicitlyWeightedMultiTaskLoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Cervical Spine Fracture Detection Demo ===")

    # 1. Setup & Configuration Overrides
    # We override Config attributes to run a fast, lightweight demo.
    print("\n[1] Configuring environment...")

    seed_everything(42)

    # Define temporary working directories
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = "./working/demo_cache"

    # Clean up previous demo runs if they exist
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)

    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Override Config
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.NUM_SLICES = 16  # Reduce depth for speed (Default 64)
    Config.IMAGE_SIZE = 128  # Reduce resolution for speed (Default 256)
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Only use 4 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(
        f"Configured: Slices={Config.NUM_SLICES}, Size={Config.IMAGE_SIZE}, Batch={Config.BATCH_SIZE}"
    )

    # 2. Data Preparation
    print("\n[2] Preparing demo metadata...")
    # Load original metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select a small subset (4 samples)
    # We ensure we pick samples that actually have directories in train_images
    valid_samples = []
    for idx, row in full_train_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        if (
            os.path.exists(full_path)
            and len(glob.glob(os.path.join(full_path, "*"))) > 0
        ):
            valid_samples.append(row)
            if len(valid_samples) >= Config.DEBUG_SAMPLE_SIZE:
                break

    if len(valid_samples) < Config.DEBUG_SAMPLE_SIZE:
        print("Warning: Not enough valid samples found. Using what is available.")

    demo_df = pd.DataFrame(valid_samples)

    # Save temporary metadata files
    demo_train_path = os.path.join(DEMO_WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(DEMO_WORKING_DIR, "demo_val.csv")

    demo_df.to_csv(demo_train_path, index=False)
    demo_df.to_csv(demo_val_path, index=False)  # Use same for val to ensure it runs

    # Point Config to these new files
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path

    print(f"Created demo metadata with {len(demo_df)} samples.")

    # 3. Preprocessing
    print("\n[3] Running Preprocessing and Caching...")
    # This will load DICOMs, window, resize, and save .npy files to DEMO_CACHE_DIR
    preprocess_and_cache(demo_df, load_cached_data=False)

    # Verify cache files exist
    cached_files = glob.glob(os.path.join(DEMO_CACHE_DIR, "*.npy"))
    print(f"Cached files found: {len(cached_files)}")
    if len(cached_files) == 0:
        raise RuntimeError("Preprocessing failed to generate cache files.")

    # 4. Dataset Verification
    print("\n[4] Verifying Dataset...")
    dataset = CervicalSpineDataset(demo_df, mode="train")

    # Fetch one item
    data_tensor, label_tensor = dataset[0]

    # Check shapes
    # Expected Data: (D, C, H, W) -> (16, 3, 128, 128)
    expected_shape = (
        Config.NUM_SLICES,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    print(f"Sample Tensor Shape: {data_tensor.shape}")
    print(f"Label Tensor Shape: {label_tensor.shape}")

    if data_tensor.shape != expected_shape:
        raise AssertionError(
            f"Dataset output shape mismatch. Expected {expected_shape}, got {data_tensor.shape}"
        )

    if label_tensor.shape != (8,):
        raise AssertionError(
            f"Label shape mismatch. Expected (8,), got {label_tensor.shape}"
        )

    print("Dataset verification passed.")

    # 5. Model & Loss Verification
    print("\n[5] Verifying Model and Loss...")
    device = get_device()
    model = CervicalMILModel(num_classes=Config.NUM_CLASSES, pretrained=False).to(
        device
    )
    criterion = ImplicitlyWeightedMultiTaskLoss()

    # Create dummy batch: (Batch, Slices, Channels, H, W)
    dummy_input = torch.randn(
        2, Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    dummy_targets = torch.randint(0, 2, (2, 8)).float().to(device)

    # Forward pass
    logits = model(dummy_input)
    print(f"Logits Shape: {logits.shape}")

    if logits.shape != (2, 8):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 8), got {logits.shape}"
        )

    # Loss calculation
    loss = criterion(logits, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss calculation resulted in NaN or negative value.")

    print("Model and Loss verification passed.")

    # 6. Training Loop Demo
    print("\n[6] Running Trainer (1 Epoch)...")
    # Initialize Trainer (it will reload metadata from Config paths we set earlier)
    trainer = Trainer(debug=True)

    # Run training
    trainer.fit(epochs=1)

    # Check if model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        # Note: It might not save if validation metric doesn't improve over infinity (which it should)
        # However, trainer logic initializes best_score = inf, so first val should save.
        print(
            "Notice: 'best_model.pth' was not found. This might happen if validation score was NaN."
        )
    else:
        print(f"Training complete. Model saved at {best_model_path}")

    # 7. Metric Calculation Verification
    print("\n[7] Verifying Metric Calculation...")
    # Create synthetic ground truth and predictions
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Case 1: Perfect predictions
    y_true_perf = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0, 1, 1]]), columns=cols)
    y_pred_perf = pd.DataFrame(
        np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.99]]), columns=cols
    )

    loss_perf = calculate_weighted_log_loss(y_true_perf, y_pred_perf)
    print(f"Metric (Near Perfect): {loss_perf:.6f}")

    # Case 2: Wrong predictions
    y_pred_bad = pd.DataFrame(
        np.array([[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.01, 0.01]]), columns=cols
    )
    loss_bad = calculate_weighted_log_loss(y_true_perf, y_pred_bad)
    print(f"Metric (Bad): {loss_bad:.6f}")

    if loss_perf >= loss_bad:
        raise AssertionError(
            "Metric logic error: Perfect prediction has higher loss than bad prediction."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
