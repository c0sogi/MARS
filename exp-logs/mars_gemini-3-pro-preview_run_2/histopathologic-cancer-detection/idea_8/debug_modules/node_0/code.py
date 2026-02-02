import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import library components
from library.config import Config
from library.utils import (
    seed_everything,
    MetricMonitor,
    ModelEMA,
    save_checkpoint,
    load_checkpoint,
)
from library.data import get_loaders
from library.models import get_model
from library.engine import train_one_epoch, validate, predict


def run_pipeline_demo():
    print("=== Starting Pathology Classification Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[Step 1] Setting up configuration for fast execution...")

    # Override Config for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small sample size for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Run in main process for debug stability
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"   Device: {Config.DEVICE}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 2] Initializing Data Loaders...")

    # load_cached_data=False forces the loader to process the debug subset
    # from metadata rather than loading a potentially large cached file.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify Train Loader Batch
    try:
        batch = next(iter(train_loader))
        images = batch["image"]
        labels = batch["label"]
        ids = batch["id"]

        print(f"   Train Batch - Image Shape: {images.shape}")
        print(f"   Train Batch - Label Shape: {labels.shape}")

        # Assertions
        expected_img_shape = (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        )
        assert (
            images.shape == expected_img_shape
        ), f"Expected image shape {expected_img_shape}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
        assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch"
        print("   -> Data Loading Verified.")

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # ---------------------------------------------------------
    # 3. Model Instantiation
    # ---------------------------------------------------------
    print("\n[Step 3] Instantiating Model...")

    model_name = "convnext_tiny"
    # pretrained=False to ensure we don't rely on external downloads for this logic check
    model = get_model(model_name, pretrained=False)
    model.to(Config.DEVICE)

    # Verify Model Output Shape
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Model Output Shape: {output.shape}")

    # timm models with num_classes=1 output (B, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output (B, 1), got {output.shape}"
    print("   -> Model Instantiation Verified.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    ema_model = ModelEMA(model, device=Config.DEVICE)

    # Train for one epoch
    avg_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        epoch=1,
        ema_model=ema_model,
    )

    print(f"   Epoch 1 Average Loss: {avg_loss:.4f}")
    assert isinstance(avg_loss, float), "Loss should be a float"
    assert avg_loss > 0, "Loss should be positive"
    print("   -> Training Loop Verified.")

    # ---------------------------------------------------------
    # 5. Validation Loop
    # ---------------------------------------------------------
    print("\n[Step 5] Running Validation...")

    # Validate using the EMA model
    val_auc = validate(ema_model.module, val_loader, Config.DEVICE)

    print(f"   Validation AUC: {val_auc:.4f}")
    assert 0.0 <= val_auc <= 1.0, f"AUC {val_auc} is out of range [0, 1]"
    print("   -> Validation Logic Verified.")

    # ---------------------------------------------------------
    # 6. Checkpointing
    # ---------------------------------------------------------
    print("\n[Step 6] Testing Checkpoint System...")

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_checkpoint.pth")
    save_checkpoint(model, optimizer, None, 1, val_auc, ckpt_path)

    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    # Reload to verify
    loaded_epoch, loaded_score = load_checkpoint(
        ckpt_path, model, optimizer, device=Config.DEVICE
    )

    assert loaded_epoch == 1, f"Expected epoch 1, got {loaded_epoch}"
    assert abs(loaded_score - val_auc) < 1e-6, "Loaded score does not match saved score"
    print("   -> Checkpointing Verified.")

    # ---------------------------------------------------------
    # 7. Inference / Prediction
    # ---------------------------------------------------------
    print("\n[Step 7] Running Inference (TTA)...")

    test_ids, test_preds = predict(ema_model.module, test_loader, Config.DEVICE)

    print(f"   Generated {len(test_preds)} predictions.")

    # Verify counts
    # Note: In debug mode, test_loader loads Config.DEBUG_SAMPLE_SIZE items.
    expected_count = min(Config.DEBUG_SAMPLE_SIZE, len(test_loader.dataset))
    assert (
        len(test_preds) == expected_count
    ), f"Expected {expected_count} predictions, got {len(test_preds)}"
    assert len(test_ids) == expected_count, "ID list length mismatch"

    # Verify probability range
    preds_np = np.array(test_preds)
    assert np.all(
        (preds_np >= 0) & (preds_np <= 1)
    ), "Predictions contain values outside [0, 1]"

    print(f"   Example Prediction: ID={test_ids[0]}, Prob={test_preds[0]:.4f}")
    print("   -> Inference Verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
