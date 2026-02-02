import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    PCT_START,
    DIV_FACTOR,
    FINAL_DIV_FACTOR,
    SUBMISSION_PATH,
    TEST_CSV,
    WORKING_DIR,
)
from library.utils import seed_everything, calculate_roc_auc, save_checkpoint
from library.dataset import get_dataloaders
from library.model import SwinTransformerGLU


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Handles the training of a single epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape [B, 1]

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, tabular)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, tabular)
            loss = criterion(logits, targets)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    epoch_loss = running_loss / dataset_size

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, auc_score


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, tabular, _ in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training(load_cached_data=True, patience=5):
    """
    Main execution function to setup data, model, and run the training loop.

    Args:
        load_cached_data (bool): Whether to use cached tabular features.
        patience (int): Early stopping patience epochs.
    """
    seed_everything()

    print(f"Starting training on device: {DEVICE}")

    # 1. Prepare Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Determine tabular input dimension from the dataset
    # We access the underlying dataset's tabular data to check shape
    # train_loader.dataset is ISICDataset
    # train_loader.dataset.tabular_data is numpy array [N, Features]
    tabular_dim = train_loader.dataset.tabular_data.shape[1]
    print(f"Tabular Feature Dimension: {tabular_dim}")

    # 2. Initialize Model
    model = SwinTransformerGLU(tabular_input_dim=tabular_dim, pretrained=True)
    model.to(DEVICE)

    # 3. Setup Training Components
    # BCEWithLogitsLoss is more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=PCT_START,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
    )

    # 4. Training Loop
    best_auc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    print("Beginning training loop...")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            best_epoch = epoch
            early_stop_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filename=f"checkpoint_epoch_{epoch}.pth",
            )
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc} at Epoch {best_epoch}")

    # 5. Inference on Test Set
    print("Loading best model for inference...")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # Load weights
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    print("Generating predictions on test set...")
    test_preds = inference(model, test_loader, DEVICE)

    # 6. Save Submission
    # Use the dataframe from the dataset to ensure alignment with predictions
    test_df = test_loader.dataset.df

    # Ensure lengths match
    if len(test_df) != len(test_preds):
        print(
            f"Warning: Length mismatch. Test DF: {len(test_df)}, Preds: {len(test_preds)}"
        )

    submission_df = pd.DataFrame(
        {"image_name": test_df["image_name"], "target": test_preds}
    )

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
