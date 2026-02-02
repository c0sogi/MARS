import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.data import get_folds, get_loaders
from library.models import HeterogeneousExpert


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates Loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # One-hot encode targets for ROC AUC calculation
    # targets shape: (N,), preds shape: (N, Num_Classes)
    targets_one_hot = np.zeros_like(preds)
    targets_one_hot[np.arange(len(targets)), targets] = 1

    try:
        # Calculate Macro-Average ROC AUC (One-vs-Rest)
        auc = roc_auc_score(targets_one_hot, preds, average="macro", multi_class="ovr")
    except ValueError:
        # Fallback if a class is missing in the batch (unlikely with stratified split)
        auc = 0.5

    if np.isnan(auc):
        auc = 0.5

    return avg_loss, auc


def fit_model(
    backbone_name,
    img_size,
    fold,
    epochs=Config.epochs,
    patience=Config.patience,
    debug=False,
    load_cached_data=True,
):
    """
    Orchestrates the training process for a specific fold and backbone.
    """
    seed_everything(Config.seed)
    device = Config.device

    # 1. Prepare Data
    # We need the training dataframe for this fold to calculate class weights
    # get_folds returns the full dataframe with 'fold' column
    full_df = get_folds(
        pd.read_csv(
            os.path.join(Config.metadata_dir, "train.csv")
        ),  # Dummy read to match sig, actual logic inside get_folds handles it or we pass it
        n_folds=Config.n_folds,
        seed=Config.seed,
        load_cached_data=load_cached_data,
    )

    # Filter for training set of this fold
    train_df_fold = full_df[full_df["fold"] != fold]

    # Calculate Class Weights
    class_weights = calculate_class_weights(
        train_df_fold, Config.target_cols, load_cached_data=load_cached_data
    )
    class_weights = class_weights.to(device)

    # Get DataLoaders
    train_loader, valid_loader = get_loaders(
        fold=fold,
        img_size=img_size,
        batch_size=Config.batch_size,
        n_folds=Config.n_folds,
        seed=Config.seed,
        num_workers=Config.num_workers,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Initialize Model
    model = HeterogeneousExpert(
        backbone_name=backbone_name, num_classes=Config.num_classes, pretrained=True
    )
    model = model.to(device)

    # 3. Setup Optimization
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.min_lr
    )
    scaler = GradScaler()

    # 4. Training Loop
    best_auc = -1.0
    patience_counter = 0

    model_filename = f"{backbone_name.replace('.', '_')}_fold_{fold}.pth"
    model_save_path = os.path.join(Config.working_dir, model_filename)

    print(f"Starting training for {backbone_name} (Fold {fold})...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_auc = validate(model, valid_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}: "
            f"Train Loss {train_loss}, "
            f"Val Loss {val_loss}, "
            f"Val AUC {val_auc}, "
            f"Time {elapsed:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model saved to {model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Clean up
    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return best_auc
