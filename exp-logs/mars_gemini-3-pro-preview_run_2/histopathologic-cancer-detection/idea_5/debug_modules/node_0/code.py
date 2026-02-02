import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MetricMonitor, calculate_auc, EarlyStopping
from library.dataset import create_datasets, get_transforms
from library.model import ConvNeXtGeM, GeMPooling
from library.engine import train_epoch, valid_epoch, inference_fn


def run_demonstration():
    print("Starting Library Demonstration...")

    # --- 1. Configuration & Setup ---
    print("\n[1] Configuring Environment for Fast Demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_DATA_LIMIT = 50  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Initialize environment
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # --- 2. Dataset & DataLoader ---
    print("\n[2] Verifying Dataset and Data Loading...")

    # Force reload to ensure we use the debug limit
    train_ds, val_ds, test_ds = create_datasets(load_cached_data=False)

    # Verify dataset sizes
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")

    # Assertions
    assert (
        len(train_ds) <= Config.DEBUG_DATA_LIMIT
    ), "Train dataset size exceeds debug limit"
    assert (
        len(val_ds) <= Config.DEBUG_DATA_LIMIT
    ), "Val dataset size exceeds debug limit"

    # Verify item shape
    sample_img, sample_label = train_ds[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    assert sample_img.shape == (
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Expected image shape (3, {Config.CROP_SIZE}, {Config.CROP_SIZE}), got {sample_img.shape}"
    assert sample_label.shape == (
        1,
    ), f"Expected label shape (1,), got {sample_label.shape}"

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,  # Ensure batch size consistency for demo
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Model Architecture ---
    print("\n[3] Verifying Model Architecture (ConvNeXtGeM)...")

    model = ConvNeXtGeM(config=Config)
    model.to(Config.DEVICE)

    # Verify Model Structure
    print(f"Backbone: {Config.MODEL_NAME}")
    print(f"Pooling: {model.gem}")

    # Test Forward Pass with Dummy Data
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape ({Config.BATCH_SIZE}, {Config.NUM_CLASSES}), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # --- 4. Training Engine ---
    print("\n[4] Demonstrating Training Loop...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one epoch
    train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE, epoch=1)

    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # --- 5. Validation Engine ---
    print("\n[5] Demonstrating Validation Loop...")

    val_loss, val_auc = valid_epoch(model, val_loader, Config.DEVICE)

    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"  # 0.5 if single class

    # --- 6. Inference Engine (TTA) ---
    print("\n[6] Demonstrating Inference with TTA...")

    # Using val_loader as test_loader for demonstration
    preds = inference_fn(model, val_loader, Config.DEVICE)

    print(f"Predictions Shape: {preds.shape}")
    print(f"First 5 Predictions: {preds[:5]}")

    assert len(preds) == len(val_ds), "Number of predictions must match dataset size"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities [0, 1]"

    # --- 7. Utilities Verification ---
    print("\n[7] Verifying Utilities...")

    # MetricMonitor
    monitor = MetricMonitor()
    monitor.update("test_metric", 10.0, n=2)
    monitor.update("test_metric", 20.0, n=2)  # Avg should be (20+40)/4 = 15
    avg = monitor.get_avg("test_metric")
    print(f"MetricMonitor Avg: {avg}")
    assert avg == 15.0, f"MetricMonitor failed, expected 15.0, got {avg}"

    # EarlyStopping
    es = EarlyStopping(patience=2, mode="max")
    es(0.5)  # Best
    assert not es.early_stop
    es(0.4)  # Worse (1)
    assert not es.early_stop
    es(0.4)  # Worse (2) -> Stop
    assert es.early_stop
    print("EarlyStopping logic verified.")

    # GeM Pooling Check
    gem = GeMPooling(p=3.0)
    t = torch.ones(1, 64, 10, 10) * 2  # Input all 2s
    out = gem(t)
    # GeM of all 2s should be 2
    assert torch.isclose(out.mean(), torch.tensor(2.0)), "GeM Pooling math check failed"
    print("GeM Pooling logic verified.")

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    run_demonstration()
