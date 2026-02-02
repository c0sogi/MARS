import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config, setup_reproducibility
from library.data import get_dataloaders
from library.model import SiameseEfficientNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for texture, context, labels in dataloader:
        texture = texture.to(device)
        context = context.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass (Siamese inputs)
        logits = model(texture, context)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * texture.size(0)

        # Store predictions (sigmoid for probability) and targets for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.extend(labels.detach().cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate AUC (handle potential edge case of single-class batch)
    # Cite debug_lesson_1: Guard Metric Calculations Against Single-Class Data Subsets
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        if np.isnan(epoch_auc):
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for texture, context, labels in dataloader:
            texture = texture.to(device)
            context = context.to(device)
            labels = labels.to(device)

            logits = model(texture, context)
            loss = criterion(logits, labels)

            running_loss += loss.item() * texture.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.extend(labels.detach().cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / len(dataloader.dataset)

    # Cite debug_lesson_1: Guard Metric Calculations Against Single-Class Data Subsets
    if len(np.unique(all_targets)) < 2:
        val_auc = 0.5
    else:
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        if np.isnan(val_auc):
            val_auc = 0.5

    return val_loss, val_auc


def run_training(load_cached=True):
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup Reproducibility and Device
    setup_reproducibility(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, _, _ = get_dataloaders(load_cached=load_cached)

    # 3. Initialize Model
    print("Initializing SiameseEfficientNet...")
    model = SiameseEfficientNet()
    model.to(device)

    # 4. Optimizer and Loss
    # Using AdamW with aggressive weight decay as per configuration
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # BCEWithLogitsLoss includes Sigmoid layer for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_val_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train Step
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Time: {epoch_duration:.2f}s")
        print(f"Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} | Val AUC: {val_auc}")

        # Early Stopping and Model Checkpointing
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved from {best_val_auc} to {val_auc}. Saving model to {best_model_path}..."
            )
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation AUC. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s.")
    print(f"Best Validation AUC: {best_val_auc}")

    return best_val_auc
