import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, EarlyStopping
from library.dataset import load_dataset
from library.model import EnsembleDenseNet
from library.engine import train_one_epoch, validate, predict_tta


def run_demo():
    print("=" * 50)
    print("Starting Tumor Detection Library Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for a fast, lightweight run
    Config.DEBUG = True
    Config.MAX_TRAIN_SAMPLES = 64  # Small subset for training
    Config.MAX_TEST_SAMPLES = 32  # Small subset for testing
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use main thread for simplicity in demo
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated. Seed set.")

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader
    # ---------------------------------------------------------
    print("\n[2] Testing Dataset Loading...")

    # Load datasets (Debug mode automatically subsamples)
    train_ds = load_dataset("train", debug=True)
    val_ds = load_dataset("val", debug=True)
    test_ds = load_dataset("test", debug=True)

    print(f"    Train Dataset Size: {len(train_ds)}")
    print(f"    Val Dataset Size:   {len(val_ds)}")
    print(f"    Test Dataset Size:  {len(test_ds)}")

    # Verify Data Loading
    img, label, img_id = train_ds[0]

    # Check Image Shape: (C, H, W) -> (3, 48, 48) based on Config.IMG_SIZE
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"

    # Check Label type
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    print("    Item retrieval verification passed.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("    DataLoaders created successfully.")

    # ---------------------------------------------------------
    # 3. Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    # Instantiate model (pretrained=False for speed in this demo)
    model = EnsembleDenseNet(arch_name="densenet121", pretrained=False, num_classes=1)
    model.to(Config.DEVICE)

    # Verify specific architectural changes
    # 1. Conv0 should be 3x3, stride 1
    conv0 = model.features.conv0
    assert conv0.kernel_size == (3, 3), "Conv0 kernel size should be 3x3"
    assert conv0.stride == (1, 1), "Conv0 stride should be 1"

    # 2. Pool0 should be Identity (no pooling)
    assert isinstance(
        model.features.pool0, torch.nn.Identity
    ), "Pool0 should be Identity"

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("    Model architecture and forward pass verified.")

    # ---------------------------------------------------------
    # 4. Training Engine
    # ---------------------------------------------------------
    print("\n[4] Testing Training Engine...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one epoch
    train_loss = train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
    )

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print(f"    Training epoch completed. Loss: {train_loss:.4f}")

    # ---------------------------------------------------------
    # 5. Validation Engine
    # ---------------------------------------------------------
    print("\n[5] Testing Validation Engine...")

    val_loss, val_auc = validate(model, val_loader, Config.DEVICE)

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"
    print(f"    Validation completed. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # ---------------------------------------------------------
    # 6. Inference / TTA
    # ---------------------------------------------------------
    print("\n[6] Testing Inference with TTA...")

    preds_df = predict_tta(model, test_loader, Config.DEVICE)

    # Verify Output DataFrame
    assert isinstance(preds_df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert (
        "id" in preds_df.columns and "label" in preds_df.columns
    ), "DataFrame must contain 'id' and 'label' columns"
    assert len(preds_df) == len(
        test_ds
    ), f"Prediction count {len(preds_df)} does not match dataset size {len(test_ds)}"

    # Check probability range
    probs = preds_df["label"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be between 0 and 1"

    print("    Inference produced valid DataFrame.")
    print(f"    Sample predictions:\n{preds_df.head(3)}")

    # ---------------------------------------------------------
    # 7. Utilities Verification
    # ---------------------------------------------------------
    print("\n[7] Testing Utilities...")

    # AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("    AverageMeter verified.")

    # EarlyStopping
    checkpoint_path = os.path.join(Config.WORKING_DIR, "test_checkpoint.pth")
    es = EarlyStopping(patience=2, mode="max", path=checkpoint_path, verbose=False)

    # Step 1: Improvement
    es(0.5, model, optimizer, epoch=1)
    assert es.best_score == 0.5
    assert es.counter == 0
    assert os.path.exists(checkpoint_path), "Checkpoint should be saved on improvement"

    # Step 2: No Improvement
    es(0.4, model, optimizer, epoch=2)
    assert es.counter == 1

    # Step 3: No Improvement (Trigger Stop)
    es(0.4, model, optimizer, epoch=3)
    assert es.counter == 2
    assert es.early_stop is True
    print("    EarlyStopping logic verified.")

    print("\n" + "=" * 50)
    print("Demo Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
