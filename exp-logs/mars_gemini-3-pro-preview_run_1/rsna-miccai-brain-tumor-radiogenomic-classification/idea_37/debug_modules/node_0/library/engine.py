import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import ARVSNet
from library.dataset import get_dataloader


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Handle edge case where only one class is present in the batch/subset
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    else:
        auc = 0.5

    return avg_loss, auc


def train_model(num_epochs=None, patience=5, max_batches_per_epoch=None):
    """
    Main training loop with Early Stopping.
    """
    if num_epochs is None:
        num_epochs = Config.NUM_EPOCHS

    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Initialize Model
    model = ARVSNet()
    model = model.to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure cache dir exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_batches_per_epoch,
        )

        val_loss, val_auc = validate(
            model, val_loader, criterion, device, max_batches=max_batches_per_epoch
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_model_path


def predict_and_submit(model_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = ARVSNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = get_dataloader("test")

    ids = []
    predictions = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten().tolist())
            ids.extend(subject_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())
