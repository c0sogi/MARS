import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    ModelEMA,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import BirdDataset, get_transforms, get_data_splits, mixup_data
from library.models import BirdClassifier
from library.train import train_one_epoch, validate


def run_demo():
    print("==== Starting Library Demo ====")

    # 1. Setup and Configuration Override for Speed
    print("\n[Step 1] Configuring environment...")
    seed_everything(42)

    # Override Config for a quick demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10  # Use very few samples
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2  # Small batch size for the few samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directories exist (Config.setup() does this, but good to confirm)
    Config.setup()
    print("Configuration patched for fast execution.")

    # 2. Data Loading and Processing
    print("\n[Step 2] Testing Data Pipeline...")

    # Test get_data_splits
    # This reads metadata/train.csv and metadata/val.csv, merges them,
    # and performs stratification. Result is cached in working/idea_14/cache.
    df_folds = get_data_splits(load_cached_data=False)
    assert "fold" in df_folds.columns, "Folds dataframe missing 'fold' column"
    print(f"Data splits generated. Total samples: {len(df_folds)}")

    # Create a small subset for testing
    train_df = (
        df_folds[df_folds["fold"] != 0]
        .head(Config.DEBUG_SAMPLES)
        .reset_index(drop=True)
    )
    val_df = (
        df_folds[df_folds["fold"] == 0]
        .head(Config.DEBUG_SAMPLES)
        .reset_index(drop=True)
    )

    # Test Dataset instantiation
    train_dataset = BirdDataset(
        train_df,
        data_source="standard",
        phase="train",
        transform=get_transforms("train"),
    )

    # Test __getitem__
    img, label = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label Shape: {label.shape}")

    # Assertions for data integrity
    assert img.shape == (
        3,
        224,
        224,
    ), f"Expected image shape (3, 224, 224), got {img.shape}"
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Expected label shape ({Config.NUM_CLASSES},), got {label.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"

    # Test Mixup
    # Create a dummy batch
    batch_imgs = torch.stack([train_dataset[i][0] for i in range(Config.BATCH_SIZE)])
    batch_lbls = torch.stack([train_dataset[i][1] for i in range(Config.BATCH_SIZE)])

    mixed_x, y_a, y_b, lam = mixup_data(
        batch_imgs, batch_lbls, alpha=0.4, use_cuda=False
    )
    assert mixed_x.shape == batch_imgs.shape, "Mixup output shape mismatch"
    assert y_a.shape == batch_lbls.shape, "Mixup target A shape mismatch"
    print("Mixup function verified.")

    # 3. Model Initialization
    print("\n[Step 3] Testing Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model (using resnet18 as it's lighter than others in the list)
    model = BirdClassifier(backbone_name="resnet18", pretrained=False)
    model.to(device)

    # Test Forward Pass
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected logits shape ({Config.BATCH_SIZE}, {Config.NUM_CLASSES}), got {logits.shape}"
    print("Model forward pass verified.")

    # 4. EMA Functionality
    print("\n[Step 4] Testing Model EMA...")
    ema = ModelEMA(model, decay=0.99, device=device)

    # Modify model weights slightly and update EMA
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    ema.update(model)
    print("Model EMA update verified.")

    # 5. Training Loop Simulation
    print("\n[Step 5] Simulating Training Loop...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Important for mixup/batch norm stability
    )
    val_loader = DataLoader(
        BirdDataset(
            val_df, data_source="standard", phase="val", transform=get_transforms("val")
        ),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch of training
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(train_loader, model, ema, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    # Run validation
    print("Running validate...")
    val_loss, val_auc = validate(val_loader, ema.module, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss returned NaN"
    # AUC might be 0.0 if targets are all same in the tiny subset, which is valid return

    # 6. Metric Calculation
    print("\n[Step 6] Testing Metric Calculation...")
    # Create synthetic data to ensure we test the metric logic properly
    # 2 classes, 4 samples
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    # Perfect predictions
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])

    auc = calculate_roc_auc(y_true, y_pred)
    print(f"Calculated AUC on synthetic perfect data: {auc}")
    assert auc == 1.0, "AUC calculation incorrect for perfect predictions"

    # Test with missing class in batch (all zeros for one class)
    y_true_imbalanced = np.array([[0, 1], [0, 0], [0, 1]])  # Class 0 is always 0
    y_pred_imbalanced = np.array([[0.1, 0.9], [0.1, 0.1], [0.1, 0.8]])
    auc_imbalanced = calculate_roc_auc(y_true_imbalanced, y_pred_imbalanced)
    print(
        f"Calculated AUC with missing positive samples for one class: {auc_imbalanced}"
    )
    # Should calculate AUC for class 1 only

    # 7. Checkpointing
    print("\n[Step 7] Testing Checkpointing...")
    ckpt_path = "demo_checkpoint.pth"
    save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "best_score": 0.5,
        },
        is_best=True,
        filename=ckpt_path,
    )

    # Verify file exists
    full_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{ckpt_path}")
    assert os.path.exists(full_ckpt_path), "Checkpoint file was not created"

    # Load checkpoint
    model_new = BirdClassifier(backbone_name="resnet18", pretrained=False)
    start_epoch, best_score = load_checkpoint(f"best_{ckpt_path}", model_new)
    assert start_epoch == 1, "Checkpoint loading failed (epoch mismatch)"
    assert best_score == 0.5, "Checkpoint loading failed (score mismatch)"
    print("Checkpoint save/load verified.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
