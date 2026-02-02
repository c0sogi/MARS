import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint, AverageMeter
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.models import CassavaModel
from library.engine import train_one_epoch, validate, inference_tta


def run_demo():
    # 1. Setup & Configuration Overrides for Speed
    print("--- Setting up environment ---")
    seed_everything(Config.SEED)

    # Override Config for faster demonstration
    Config.IMG_SIZE = 224  # Reduce size from 384 for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 1

    # Use a lightweight model for the demo instead of the heavy ViT/BEiT
    DEMO_MODEL_NAME = "resnet18"

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading (Subsets)
    print("\n--- Loading Metadata & Creating Subsets ---")
    # Load metadata
    df_train_full = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_full = pd.read_csv(Config.VAL_META_PATH)
    df_test_full = pd.read_csv(Config.TEST_META_PATH)

    # Take a tiny subset (e.g., 16 samples) to simulate a quick epoch
    df_train_demo = df_train_full.head(16).reset_index(drop=True)
    df_val_demo = df_val_full.head(8).reset_index(drop=True)
    df_test_demo = df_test_full.head(8).reset_index(drop=True)

    print(f"Train subset shape: {df_train_demo.shape}")
    print(f"Val subset shape: {df_val_demo.shape}")

    # 3. Dataset & Transforms
    print("\n--- Initializing Datasets ---")
    train_dataset = CassavaDataset(
        df_train_demo, transforms=get_transforms(data_type="train"), output_label=True
    )
    val_dataset = CassavaDataset(
        df_val_demo, transforms=get_transforms(data_type="val"), output_label=True
    )
    test_dataset = CassavaDataset(
        df_test_demo,
        transforms=get_transforms(data_type="test"),
        output_label=False,  # Test set usually doesn't have labels for inference
    )

    # Verification: Check __getitem__
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image must be a torch.Tensor"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
    print("Dataset verification passed.")

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 5. Mixup Verification
    print("\n--- Verifying Mixup Logic ---")
    mixup_fn = Mixup(
        prob=1.0, mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=Config.NUM_CLASSES
    )

    # Get a batch
    dummy_imgs, dummy_targets = next(iter(train_loader))
    dummy_imgs, dummy_targets = dummy_imgs.to(device), dummy_targets.to(device)

    # Apply Mixup
    mixed_imgs, mixed_targets = mixup_fn(dummy_imgs, dummy_targets)

    # Assertions
    assert (
        mixed_imgs.shape == dummy_imgs.shape
    ), "Mixup should not change image dimensions"
    assert mixed_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Mixup targets should be one-hot encoded [B, Num_Classes]"
    assert mixed_targets.dtype == torch.float32, "Mixup targets should be float"
    print("Mixup verification passed.")

    # 6. Model Initialization
    print(f"\n--- Initializing Model: {DEMO_MODEL_NAME} ---")
    model = CassavaModel(
        model_name=DEMO_MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    # Dummy Forward Pass
    with torch.no_grad():
        dummy_out = model(dummy_imgs)
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model forward pass verification passed.")

    # 7. Training Engine
    print("\n--- Running Training Engine (1 Epoch) ---")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Run train_one_epoch
    # Note: We use the mixup_fn verified earlier
    avg_loss = train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        mixup_fn=mixup_fn,
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN"
    assert avg_loss > 0, "Training loss should be positive"
    print(f"Training finished. Average Loss: {avg_loss:.4f}")

    # 8. Validation Engine
    print("\n--- Running Validation Engine ---")
    val_acc = validate(model, val_loader, device)
    assert 0 <= val_acc <= 100, "Validation accuracy should be between 0 and 100"
    print(f"Validation finished. Accuracy: {val_acc:.2f}%")

    # 9. Inference TTA
    print("\n--- Running Inference TTA ---")
    # inference_tta expects a model and a loader
    preds = inference_tta(model, test_loader, device)

    assert preds.shape == (
        len(df_test_demo),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch"
    # Check if probabilities sum roughly to 1 (softmax applied in TTA)
    sums = preds.sum(dim=1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ), "Probabilities do not sum to 1"
    print("Inference TTA verification passed.")

    # 10. Utils: Logger and Checkpoint
    print("\n--- Verifying Utils ---")
    # Logger
    log_path = os.path.join(Config.OUTPUT_DIR, "demo.log")
    logger = get_logger(log_path)
    logger.info("This is a test log entry.")
    assert os.path.exists(log_path), "Log file was not created"

    # Checkpoint
    ckpt_state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(
        ckpt_state, is_best=True, output_dir=Config.OUTPUT_DIR, filename="demo_ckpt.pth"
    )
    assert os.path.exists(
        os.path.join(Config.OUTPUT_DIR, "demo_ckpt.pth")
    ), "Checkpoint file missing"
    assert os.path.exists(
        os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    ), "Best model copy missing"
    print("Utils verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
