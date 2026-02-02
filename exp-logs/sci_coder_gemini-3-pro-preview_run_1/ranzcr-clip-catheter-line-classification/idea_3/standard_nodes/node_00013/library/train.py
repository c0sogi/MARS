import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from timm.utils import ModelEmaV2
from library.config import Config
from library.utils import seed_everything, get_score, get_pos_weights
from library.dataset import get_dataloaders
from library.model import CatheterModel


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, ema=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=Config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)

        # Apply sigmoid to logits for scoring
        probs = torch.sigmoid(logits)

        y_true_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    epoch_auc = get_score(y_true, y_pred)

    return epoch_loss, epoch_auc


def valid_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)

            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    epoch_auc = get_score(y_true, y_pred)

    return epoch_loss, epoch_auc


def run_training(
    debug=Config.debug,
    num_epochs=Config.num_epochs,
    batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    weight_decay=Config.weight_decay,
    patience=Config.patience,
    model_save_path=Config.model_save_path,
    pos_weights_path=Config.pos_weights_path,
):
    """
    Main training function.
    """
    seed_everything(Config.seed)
    device = Config.device

    print(f"Device: {device}")
    print(f"Model: {Config.model_name}")
    print(f"Image Size: {Config.image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    if debug:
        print(f"Debug mode: Sampling {Config.debug_sample_size} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        # We don't necessarily need to sample test_df for training loop, but good for consistency if full pipeline
        test_df = test_df.sample(
            n=min(len(test_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # 2. Prepare DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size
    )

    # 3. Calculate Class Weights
    # We calculate weights based on the training dataframe
    pos_weights = get_pos_weights(
        Config.target_cols,
        df=train_df,
        file_path=pos_weights_path,
        load_cached_data=True,
    )
    pos_weights = pos_weights.to(device)

    # 4. Initialize Model, Loss, Optimizer, Scaler, EMA
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
        in_channels=Config.in_channels,
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    # EMA
    ema = None
    if Config.use_ema:
        print(f"Initializing EMA with decay {Config.ema_decay}")
        ema = ModelEmaV2(model, decay=Config.ema_decay)

    # 5. Training Loop
    best_auc = -1.0
    epochs_no_improve = 0

    print("\nStarting training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ema
        )

        # Validate using EMA model if available, otherwise standard model
        val_model = ema.module if ema else model
        val_loss, val_auc = valid_one_epoch(val_model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"  Valid Loss: {val_loss} | Valid AUC: {val_auc}")

        # Checkpoint and Early Stopping
        if val_auc > best_auc + Config.min_delta:
            print(
                f"  Validation AUC improved from {best_auc} to {val_auc}. Saving model to {model_save_path}..."
            )
            best_auc = val_auc
            # Save the model that was used for validation (EMA or Standard)
            save_model = ema.module if ema else model
            torch.save(save_model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"  No improvement in validation AUC. Patience: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\nTraining finished. Best Validation AUC: {best_auc}")
    return best_auc
