import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config, seed_everything
from library.dataset import (
    process_data,
    HotelDataset,
    get_transforms,
    BalanceClassSampler,
)
from library.model import HotelIdModel, train_one_epoch, validate, inference


def train_fn(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch using the ArcFace loss and AdamW optimizer.
    Wraps the library function to maintain modularity.
    """
    return train_one_epoch(
        model, loader, criterion, optimizer, scheduler, device, epoch
    )


def eval_fn(model, loader, device, num_classes):
    """
    Evaluates the model on the validation set using MAP@5.
    Wraps the library function to maintain modularity.
    """
    return validate(model, loader, device, num_classes)


def run(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main execution function for the training and inference pipeline.

    Args:
        epochs (int): Number of training epochs.
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Update Config with runtime arguments
    Config.EPOCHS = epochs
    Config.DEBUG = debug

    print(f"Starting run with EPOCHS={epochs}, DEBUG={debug}")

    # 1. Data Preparation
    # process_data handles caching internally (loading from .npy if available)
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=True)

    # Load Label Encoder classes for decoding predictions later
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)

    # Initialize Datasets
    train_dataset = HotelDataset(
        train_df, transforms=get_transforms("train"), root_dir=Config.INPUT_DIR
    )
    val_dataset = HotelDataset(
        val_df, transforms=get_transforms("val"), root_dir=Config.INPUT_DIR
    )
    test_dataset = HotelDataset(
        test_df,
        transforms=get_transforms("test"),
        root_dir=Config.INPUT_DIR,
        is_test=True,
    )

    # Initialize Sampler and DataLoaders
    # Class-Balanced Sampler to handle long-tail distribution
    classes_per_batch = min(Config.CLASSES_PER_BATCH, num_classes)
    train_sampler = BalanceClassSampler(
        train_df["label"].values,
        classes_per_batch=classes_per_batch,
        samples_per_class=Config.SAMPLES_PER_CLASS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    device = Config.DEVICE
    model = HotelIdModel(num_classes=num_classes).to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.MIN_LR
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_fn(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )

        # Validate
        val_map = eval_fn(model, val_loader, device, num_classes)

        # Log metrics with full precision
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Val MAP@5: {val_map}")

        # Checkpointing & Early Stopping
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best MAP@5: {best_map}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Starting inference on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using current model state.")

    # Generate predictions (returns list of space-delimited strings)
    predictions = inference(model, test_loader, device, label_encoder_classes)

    # 6. Submission Generation
    submission_df = test_df[["image"]].copy()
    submission_df["hotel_id"] = predictions

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
