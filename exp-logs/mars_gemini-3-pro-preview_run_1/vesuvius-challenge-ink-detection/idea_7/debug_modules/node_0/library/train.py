import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, dataset, model, utils


def train_epoch(loader, model, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(volumes)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * volumes.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates probabilities and targets to calculate F0.5 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            outputs = model(volumes)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * volumes.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_pred_probs = np.concatenate(all_preds)

    # Calculate F0.5 score using a default threshold of 0.5 for monitoring
    y_pred_bin = (y_pred_probs >= 0.5).astype(np.uint8)
    val_f05 = utils.calculate_fbeta(y_true, y_pred_bin, beta=0.5)

    return val_loss, val_f05


def run_training(num_epochs=config.NUM_EPOCHS, patience=3, load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    # Load Datasets
    train_loader, val_loader, _ = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # Initialize Model
    net = model.HDNet().to(config.DEVICE)

    # Define Loss and Optimizer
    # Using weighted BCEWithLogitsLoss to handle class imbalance
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Training State
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on {config.DEVICE}...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(train_loader, net, criterion, optimizer, config.DEVICE)

        # Validate
        val_loss, val_f05 = validate(val_loader, net, criterion, config.DEVICE)

        # Print metrics (full precision for F0.5 as requested)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F0.5: {val_f05}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")
