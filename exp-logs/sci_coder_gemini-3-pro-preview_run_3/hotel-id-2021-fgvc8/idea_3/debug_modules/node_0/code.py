import sys
import os
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# -------------------------------------------------------------------------
# 1. Mock tqdm to suppress output (Must be done before library imports)
# -------------------------------------------------------------------------
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = MockTqdm
sys.modules["tqdm"] = tqdm_module

# -------------------------------------------------------------------------
# 2. Import Library Components
# -------------------------------------------------------------------------
# Ensure the current directory is in path
sys.path.append(".")

from library.config import Config, seed_everything
from library.dataset import (
    process_data,
    HotelDataset,
    get_transforms,
    BalanceClassSampler,
)
from library.model import HotelIdModel, train_one_epoch, validate, inference
from library.utils import mapk


# -------------------------------------------------------------------------
# 3. Main Demonstration Script
# -------------------------------------------------------------------------
def run_demo():
    print("Initializing Hotel ID Demo...")

    # --- Configuration Overrides for Speed ---
    seed_everything(42)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small subset for instant execution
    Config.IMAGE_SIZE = 128  # Small images
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
    Config.CLASSES_PER_BATCH = 2  # Minimal classes per batch
    Config.SAMPLES_PER_CLASS = 4  # Minimal samples per class
    Config.BATCH_SIZE = Config.CLASSES_PER_BATCH * Config.SAMPLES_PER_CLASS  # 8
    Config.BACKBONE_NAME = "resnet18"  # Lightweight backbone
    Config.WORKING_DIR = "./working/demo"
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Data Processing ---
    print("Processing Data...")
    # load_cached_data=False forces the encoder to fit on our small debug subset
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=False)

    # Logic Verification
    assert len(train_df) <= Config.DEBUG_SAMPLE_SIZE
    assert num_classes > 0
    assert "label" in train_df.columns
    print(
        f"Data ready. Train: {len(train_df)}, Val: {len(val_df)}, Classes: {num_classes}"
    )

    # --- Dataset & DataLoader ---
    print("Setting up Datasets and Loaders...")
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = HotelDataset(train_df, transforms=train_transforms)
    val_dataset = HotelDataset(val_df, transforms=val_transforms)

    # Adjust classes per batch if debug set has fewer classes than config
    actual_classes = train_df["label"].nunique()
    cpb = min(Config.CLASSES_PER_BATCH, actual_classes)

    train_sampler = BalanceClassSampler(
        train_df["label"].values,
        classes_per_batch=cpb,
        samples_per_class=Config.SAMPLES_PER_CLASS,
    )

    # Note: Using sampler=train_sampler with a defined batch_size.
    # The sampler yields indices one by one (shuffled within blocks of P*K).
    # DataLoader collects them into batches of batch_size.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cpb * Config.SAMPLES_PER_CLASS,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Shapes
    images, labels = next(iter(train_loader))
    assert images.shape == (
        cpb * Config.SAMPLES_PER_CLASS,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert labels.shape == (cpb * Config.SAMPLES_PER_CLASS,)
    print("Batch shapes verified.")

    # --- Model Initialization ---
    print("Initializing Model...")
    model = HotelIdModel(
        num_classes=num_classes, backbone_name=Config.BACKBONE_NAME, pretrained=False
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        # ArcFace head requires labels for training forward pass
        outputs = model(images.to(Config.DEVICE), labels.to(Config.DEVICE))
        assert outputs.shape == (images.shape[0], num_classes)
    print("Model forward pass verified.")

    # --- Training Step ---
    print("Executing Training Step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Run one epoch (iterates the small debug loader)
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, None, Config.DEVICE, 1
    )
    assert not np.isnan(train_loss)
    print(f"Training step complete. Loss: {train_loss:.4f}")

    # --- Validation ---
    print("Executing Validation...")
    # Verify Metric Logic first
    dummy_targets = [[1], [2]]
    dummy_preds = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # 1 hit, 1 miss
    score = mapk(dummy_targets, dummy_preds, k=5)
    assert score == 0.5, f"Metric logic check failed, expected 0.5, got {score}"

    # Run actual validation
    val_map = validate(model, val_loader, Config.DEVICE, num_classes)
    print(f"Validation MAP@5: {val_map:.4f}")

    # --- Inference ---
    print("Executing Inference...")
    test_dataset = HotelDataset(
        test_df, transforms=get_transforms("test"), is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)

    # Load the encoder classes saved by process_data
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")
    le_classes = np.load(encoder_path, allow_pickle=True)

    predictions = inference(model, test_loader, Config.DEVICE, le_classes)

    # Verify Predictions
    assert len(predictions) == len(test_df)
    assert isinstance(predictions[0], str)
    # Check format: space-delimited integers
    parts = predictions[0].split()
    assert len(parts) <= Config.TOP_K
    print(f"Inference complete. Example: {predictions[0]}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
