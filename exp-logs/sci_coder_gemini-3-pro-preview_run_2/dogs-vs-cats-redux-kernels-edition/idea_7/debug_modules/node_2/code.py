import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, AverageMeter, mixup_data
from library.dataset import get_datasets
from library.models import get_model
from library.engine import fit
from library.inference import run_inference


def run_demo():
    print("=== Starting Demonstration of Library Components ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for speed...")

    # Override Config values to run a tiny, fast experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small number of samples for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.N_FOLDS = 1  # Only run 1 fold
    Config.TTA_FLIP = True  # Keep TTA on to verify logic

    # Use a lightweight model for demonstration (ResNet18) instead of the larger ConvNeXt/Swin
    Config.MODELS = ["resnet18"]

    # Setup temporary working directories
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated for fast execution.")

    # ---------------------------------------------------------
    # 2. Testing Utilities
    # ---------------------------------------------------------
    print("\n[2] Testing Utilities (library.utils)...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # Total sum = 10*2 + 20*1 = 40. Total count = 3. Avg = 13.33...
    assert meter.count == 3
    assert abs(meter.avg - 13.3333) < 1e-4
    print("AverageMeter verified.")

    # Test Mixup
    # Create dummy tensors
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, device="cpu")

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0 <= lam <= 1
    print("Mixup verified.")

    # ---------------------------------------------------------
    # 3. Testing Dataset
    # ---------------------------------------------------------
    print("\n[3] Testing Dataset (library.dataset)...")

    # Load datasets in debug mode
    train_ds, val_ds, test_ds = get_datasets(
        debug=True, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    assert len(train_ds) == Config.DEBUG_SAMPLE_SIZE

    # Check train item structure
    img, label = train_ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert label.ndim == 0  # Label should be scalar

    # Check test item structure
    img_test, img_id = test_ds[0]
    assert isinstance(img_test, torch.Tensor)
    assert isinstance(img_id, (int, np.integer))

    print("Datasets verified.")

    # ---------------------------------------------------------
    # 4. Testing Model
    # ---------------------------------------------------------
    print("\n[4] Testing Model (library.models)...")

    # Instantiate model (pretrained=False for speed/offline safety)
    model = get_model(Config.MODELS[0], pretrained=False, num_classes=1)
    model.to(Config.DEVICE)

    # Forward pass check
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
    output = model(dummy_input)

    # Output should be (B, 1) for binary classification
    assert output.shape == (2, 1)
    print("Model instantiation and forward pass verified.")

    # ---------------------------------------------------------
    # 5. Testing Engine (Training Loop)
    # ---------------------------------------------------------
    print("\n[5] Testing Engine (library.engine)...")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Construct checkpoint path matching what inference expects: "{model_name}_fold_{fold}.pth"
    fold = 0
    ckpt_name = f"{Config.MODELS[0]}_fold_{fold}.pth"
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

    print(f"Training for {Config.EPOCHS} epoch(s)...")
    # fit() runs train_one_epoch and validate, and saves the best model
    best_loss = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device=Config.DEVICE,
        checkpoint_path=ckpt_path,
    )

    assert os.path.exists(ckpt_path)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")

    # ---------------------------------------------------------
    # 6. Testing Inference
    # ---------------------------------------------------------
    print("\n[6] Testing Inference (library.inference)...")

    # run_inference iterates over Config.MODELS and Config.N_FOLDS, loads checkpoints, and generates submission
    run_inference(debug=True, debug_sample_size=Config.DEBUG_SAMPLE_SIZE)

    assert os.path.exists(Config.SUBMISSION_PATH)

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["id", "label"]
    assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE

    # Check probability value ranges
    assert sub_df["label"].min() >= 0.0
    assert sub_df["label"].max() <= 1.0

    print("Inference verified. Submission file generated correctly.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
