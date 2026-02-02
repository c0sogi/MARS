import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library import config
from library import utils
from library import data
from library import model as model_lib


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Ensure labels are (B, 1) to match model output
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluation loop. Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Ensure labels are (B, 1)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Handle edge case where only one class is present in the batch/subset
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.5

    return total_loss, auc_score


def run_training(debug=False, load_cached_data=True):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # Ensure output directory exists
    os.makedirs(config.IDEA_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_dataset, val_dataset = data.get_train_val_datasets(
        load_cached_data=load_cached_data, debug=debug
    )

    train_loader, val_loader = data.get_dataloaders(
        train_dataset, val_dataset, batch_size=config.BATCH_SIZE
    )

    # 3. Model Initialization
    print("Initializing WIVE Network...")
    net = model_lib.WIVENet()
    net = net.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.IDEA_DIR, "best_model.pth")

    epochs = config.EPOCHS

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_val_auc}")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    return best_val_auc
