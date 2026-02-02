import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import calculate_roc_auc, ModelEMA


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, device, ema_model=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if Config.USE_MIXUP:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if ema_model:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        epoch_auc = calculate_roc_auc(all_labels, all_preds)
    else:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    fold_idx,
    num_epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Orchestrates the training for a single fold, including EMA and Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()

    # Initialize EMA if configured
    ema_model = None
    if Config.USE_EMA:
        print(f"Initializing EMA with decay {Config.EMA_DECAY}")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    best_auc = -1.0
    early_stopping_counter = 0

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{model.backbone_name}_fold_{fold_idx}_best.pth"
    )

    print(f"Starting training for Fold {fold_idx} with backbone {model.backbone_name}")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ema_model
        )

        # Validate
        # Use EMA shadow model for validation if available
        val_model_to_use = ema_model.shadow if ema_model else model
        val_loss, val_auc = validate(val_model_to_use, val_loader, criterion, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed}s - "
            f"Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            early_stopping_counter = 0

            # Save best model
            save_model = ema_model.shadow if ema_model else model
            torch.save(save_model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} finished. Best Val AUC: {best_auc}")

    # Load best weights to return
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return model, best_auc


def predict(model, loader, device):
    """
    Generates predictions for a given loader.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_ids.append(rec_ids.numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_ids = np.concatenate(all_ids)
    else:
        all_preds = np.array([])
        all_ids = np.array([])

    return all_ids, all_preds
