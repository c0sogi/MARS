import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, get_logger, accuracy
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.model import CassavaModel, ModelEMA
from library.engine import train_one_epoch, valid_one_epoch


def demo_main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Setting up Configuration ---")

    # Override CFG defaults for a fast demonstration
    CFG.debug = True
    CFG.pretrained = False  # Disable weight download to ensure offline execution speed
    CFG.epochs = 1
    CFG.batch_size = 4  # Small batch size for the demo
    CFG.num_workers = 0  # Use main thread to avoid multiprocessing overhead in demo
    CFG.model_name = "resnet18"  # Use a lightweight backbone for this test

    # Set device
    device = torch.device(CFG.device)
    print(f"Running on device: {device}")

    # Ensure reproducibility
    seed_everything(CFG.seed)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("\n--- 2. Preparing Data ---")

    # Verify metadata exists
    if not os.path.exists(CFG.train_metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {CFG.train_metadata_path}")

    # Load metadata and create a small subset
    df = pd.read_csv(CFG.train_metadata_path)
    subset_size = 20
    df_subset = df.head(subset_size).copy()

    # Split into Train (16) and Validation (4)
    train_len = int(0.8 * subset_size)
    train_df = df_subset.iloc[:train_len].reset_index(drop=True)
    val_df = df_subset.iloc[train_len:].reset_index(drop=True)

    print(f"Train subset size: {len(train_df)}")
    print(f"Valid subset size: {len(val_df)}")

    # Instantiate Datasets
    # Using 224x224 resolution for speed
    img_size = 224
    train_dataset = CassavaDataset(
        train_df, transform=get_transforms("train", img_size), output_label=True
    )
    val_dataset = CassavaDataset(
        val_df, transform=get_transforms("valid", img_size), output_label=True
    )

    # Verify single item loading
    img, label = train_dataset[0]
    assert img.shape == (3, img_size, img_size), f"Image shape mismatch: {img.shape}"
    assert isinstance(label, (int, np.integer)), f"Label type mismatch: {type(label)}"
    print("Dataset item verification passed.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- 3. Initializing Model ---")

    # Initialize model with overridden name and no pretraining
    model = CassavaModel(model_name=CFG.model_name, pretrained=False)
    model.to(device)

    # Verify forward pass with dummy input
    dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        CFG.num_classes,
    ), f"Model output shape mismatch. Expected (2, {CFG.num_classes}), got {output.shape}"
    print("Model forward pass verification passed.")

    # Initialize EMA
    model_ema = ModelEMA(model)
    print("ModelEMA initialized.")

    # ==========================================
    # 4. Augmentation (Mixup) Logic
    # ==========================================
    print("\n--- 4. Testing Mixup ---")

    # Initialize Mixup with high probability for testing
    mixup_fn = Mixup(prob=1.0, switch_prob=0.5, num_classes=CFG.num_classes)

    # Fetch a batch
    imgs, targets = next(iter(train_loader))
    imgs, targets = imgs.to(device), targets.to(device)

    # Apply Mixup
    mixed_imgs, mixed_targets = mixup_fn(imgs, targets)

    # Verify shapes
    assert mixed_imgs.shape == imgs.shape, "Mixed image shape mismatch"
    # Mixed targets should be [Batch, NumClasses] (one-hot/soft)
    assert mixed_targets.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), f"Mixed target shape mismatch: {mixed_targets.shape}"

    print("Mixup augmentation verification passed.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n--- 5. Running Training Loop (1 Epoch) ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Execute one training epoch
    train_loss, train_acc = train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        scheduler=None,
        mixup_fn=mixup_fn,
        model_ema=model_ema,
    )

    # Verify metrics
    assert not np.isnan(train_loss), "Training loss resulted in NaN"
    assert 0 <= train_acc <= 100, "Training accuracy out of bounds"
    print(f"Train Epoch Completed. Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # ==========================================
    # 6. Validation Loop Demonstration
    # ==========================================
    print("\n--- 6. Running Validation Loop ---")

    # Execute validation on the main model
    val_loss, val_acc = valid_one_epoch(
        epoch=1, model=model, val_loader=val_loader, device=device
    )

    assert not np.isnan(val_loss), "Validation loss resulted in NaN"
    print(f"Validation Completed. Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Execute validation on the EMA model (optional check)
    print("Validating EMA model...")
    ema_loss, ema_acc = valid_one_epoch(
        epoch=1, model=model_ema.ema, val_loader=val_loader, device=device
    )
    print(f"EMA Validation Completed. Loss: {ema_loss:.4f}, Acc: {ema_acc:.2f}%")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    demo_main()
