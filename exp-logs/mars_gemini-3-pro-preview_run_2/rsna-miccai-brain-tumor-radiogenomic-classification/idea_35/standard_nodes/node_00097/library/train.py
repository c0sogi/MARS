import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

import library.config as C
import library.utils as U
import library.data as D
import library.model as M


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    if len(all_labels) > 0:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Handle edge case where only one class is present in batch/set
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_probs)
        else:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def fit(
    epochs=C.NUM_EPOCHS,
    batch_size=C.BATCH_SIZE,
    learning_rate=C.LEARNING_RATE,
    weight_decay=C.WEIGHT_DECAY,
    debug_limit=None,
    load_cached_data=True,
):
    """
    Main training loop with early stopping and model checkpointing.
    """
    # 1. Setup
    U.seed_everything(C.SEED)
    device = torch.device(C.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Metadata
    if not os.path.exists(C.TRAIN_METADATA_PATH) or not os.path.exists(
        C.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_csv(C.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(C.VAL_METADATA_PATH)

    print(f"Metadata loaded. Train: {len(train_df)}, Val: {len(val_df)}")

    # 3. Prepare Datasets & Dataloaders
    # Using the caching logic implemented in MGMTDataset via library.data
    train_dataset = D.MGMTDataset(
        metadata_df=train_df,
        transform=D.get_transforms(phase="train"),
        cache_path=C.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_test=False,
        debug_limit=debug_limit,
    )

    val_dataset = D.MGMTDataset(
        metadata_df=val_df,
        transform=D.get_transforms(phase="val"),
        cache_path=C.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_test=False,
        debug_limit=debug_limit,
    )

    train_loader = D.get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=C.NUM_WORKERS
    )
    val_loader = D.get_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=C.NUM_WORKERS
    )

    # 4. Initialize Model, Loss, Optimizer
    model = M.AsymmetricEfficientNet()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc + C.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), C.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= C.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
