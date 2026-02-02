import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import BreastCancerMILModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move data to device
        images = batch["images"].to(device)  # (B, V, C, H, W)
        mask = batch["mask"].to(device)  # (B, V)
        metadata = batch["metadata"].to(device)  # (B, Meta_Dim)
        labels = batch["labels"].to(device)  # (B, 1)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images, mask, metadata)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)

            batch_size = images.size(0)

            # Forward pass
            logits = model(images, mask, metadata)

            # Compute loss
            loss = criterion(logits, labels)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store for metrics
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Statistics
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate pF1 score
    pf1 = probabilistic_f1(all_labels, all_probs)

    return avg_loss, pf1


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    print("Initializing Model...")
    model = BreastCancerMILModel(config=Config)
    model.to(device)

    # 4. Loss and Optimizer
    # Handle class imbalance with pos_weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val pF1: {val_pf1}")

        # Early Stopping & Checkpointing
        if val_pf1 > best_pf1:
            print(
                f"Validation pF1 improved from {best_pf1} to {val_pf1}. Saving model..."
            )
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Load Best Model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    return model
