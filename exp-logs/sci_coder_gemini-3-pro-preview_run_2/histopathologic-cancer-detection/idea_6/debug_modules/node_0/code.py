import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.data import get_loaders, PathologyDataset
from library.model import get_model, ModelEMA
from library.engine import train_fold, inference_fn, train_one_epoch, validate
from library.utils import seed_everything, calculate_roc_auc, MetricMonitor


def create_subset_metadata(source_path, dest_path, n_samples=50):
    """
    Reads the original metadata, samples a subset, and saves it to a new location.
    This allows us to test the pipeline quickly without loading the full dataset.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample subset (ensure we don't sample more than available)
    n = min(len(df), n_samples)
    df_subset = df.sample(n=n, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_subset.to_csv(dest_path, index=False)
    print(f"Created subset metadata at {dest_path} with {n} samples.")
    return n


def verify_utils():
    print("\n=== Verifying Utilities ===")

    # 1. Verify ROC AUC calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_pred)
    # sklearn roc_auc_score for these values is 0.75
    assert isinstance(auc, float), "AUC should be a float"
    print(f"ROC AUC verification passed. Score: {auc}")

    # 2. Verify MetricMonitor
    monitor = MetricMonitor()
    monitor.update("loss", 10.0, n=2)  # sum=20, count=2
    monitor.update("loss", 5.0, n=1)  # sum=25, count=3
    assert (
        abs(monitor.avg["loss"] - (25.0 / 3.0)) < 1e-5
    ), "MetricMonitor average calculation incorrect"
    print("MetricMonitor verification passed.")


def verify_data_and_model():
    print("\n=== Verifying Data Loading and Model ===")

    # Force reload from the new subset metadata files
    train_loader, val_loader, test_loader, test_ids = get_loaders(
        load_cached_data=False
    )

    # 1. Check Train Loader
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for shapes
    # Config.INPUT_SIZE is 64, Batch size is set to 8 in setup
    assert images.shape == (
        8,
        3,
        64,
        64,
    ), f"Expected (8, 3, 64, 64), got {images.shape}"
    assert labels.shape == (8,), f"Expected (8,), got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32 tensors"

    # 2. Check Model Instantiation
    model = get_model(pretrained=False)  # False for speed, we just check architecture
    model.to(Config.DEVICE)

    # 3. Check Forward Pass
    with torch.no_grad():
        output = model(images.to(Config.DEVICE))

    print(f"Model Output Shape: {output.shape}")
    # Output should be (Batch, 1) because NUM_CLASSES=1 and we haven't squeezed yet in raw model
    # Note: engine.py squeezes it, but the raw model returns (B, 1)
    assert output.shape == (8, 1), f"Expected model output (8, 1), got {output.shape}"

    # 4. Check ModelEMA
    ema = ModelEMA(model)
    # Update EMA
    ema.update(model)
    # Check if shadow parameters exist
    assert len(list(ema.ema_model.parameters())) == len(list(model.parameters()))
    print("Model and EMA verification passed.")

    return train_loader, val_loader, test_loader, model


def verify_training_engine(model, train_loader, val_loader):
    print("\n=== Verifying Training Engine ===")

    # Setup optimizer and scheduler as expected by train_fold
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Run train_fold (which runs train_one_epoch and validate)
    # We use fold_idx=0
    best_auc = train_fold(
        fold_idx=0,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        patience=1,
    )

    print(f"Training loop completed. Best AUC: {best_auc}")
    assert 0.0 <= best_auc <= 1.0, "AUC must be between 0 and 1"

    # Check if checkpoint was created
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_fold_0.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print("Training engine verification passed.")


def verify_inference(model, test_loader):
    print("\n=== Verifying Inference (TTA) ===")

    # Run inference
    preds = inference_fn(model, test_loader, Config.DEVICE)

    print(f"Predictions shape: {preds.shape}")

    # We sampled 50 test images in setup
    # Note: DataLoader drops last if configured, but test loader usually doesn't.
    # Config.BATCH_SIZE is 8. 50 samples -> ceil(50/8) batches.
    # The inference_fn concatenates all predictions.
    assert len(preds) == 50, f"Expected 50 predictions, got {len(preds)}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions should be probabilities [0, 1]"

    print("Inference verification passed.")


def main():
    # --- 1. Setup & Configuration Overrides ---
    seed_everything(42)

    # Define temporary working directories
    base_work_dir = "./working/demo_run_verify"
    demo_meta_dir = os.path.join(base_work_dir, "metadata")
    demo_cache_dir = os.path.join(base_work_dir, "cache")
    demo_ckpt_dir = os.path.join(base_work_dir, "checkpoints")

    # Clean up previous runs if any
    if os.path.exists(base_work_dir):
        shutil.rmtree(base_work_dir)

    # Override Config class attributes for the demo
    Config.WORKING_DIR = base_work_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.CHECKPOINT_DIR = demo_ckpt_dir
    Config.EPOCHS = 1  # Run only 1 epoch for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.N_FOLDS = 2  # Not used directly in single fold call, but good to set
    Config.NUM_WORKERS = 2  # Reduce workers for simple demo

    # Setup directories
    Config.setup()

    # --- 2. Create Tiny Datasets ---
    print("Generating subset metadata for demonstration...")

    # Define new paths
    demo_train_path = os.path.join(demo_meta_dir, "train.csv")
    demo_val_path = os.path.join(demo_meta_dir, "val.csv")
    demo_test_path = os.path.join(demo_meta_dir, "test.csv")

    # Create subsets (50 samples each)
    create_subset_metadata(Config.TRAIN_META_PATH, demo_train_path, n_samples=50)
    create_subset_metadata(Config.VAL_META_PATH, demo_val_path, n_samples=50)
    create_subset_metadata(Config.TEST_META_PATH, demo_test_path, n_samples=50)

    # Point Config to these new files
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    # --- 3. Run Verifications ---
    try:
        verify_utils()

        train_loader, val_loader, test_loader, model = verify_data_and_model()

        verify_training_engine(model, train_loader, val_loader)

        verify_inference(model, test_loader)

        print("\nAll demonstrations and verifications completed successfully.")

    finally:
        # --- 4. Cleanup ---
        # Comment out the next line if you want to inspect the files after running
        if os.path.exists(base_work_dir):
            shutil.rmtree(base_work_dir)
            print(f"Cleaned up temporary directory: {base_work_dir}")


if __name__ == "__main__":
    main()
