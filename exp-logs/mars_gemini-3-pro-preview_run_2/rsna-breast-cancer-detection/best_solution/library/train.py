import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import HybridEfficientNet


def train_fn(model, loader, criterion, optimizer, scheduler, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, tab_features, targets in loader:
        batch_size = images.size(0)

        images = images.to(device)
        tab_features = tab_features.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        logits = model(images, tab_features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns validation loss and pF1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, tab_features, targets in loader:
            batch_size = images.size(0)

            images = images.to(device)
            tab_features = tab_features.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, tab_features)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate pF1 score
    # Flatten arrays to ensure 1D for metric calculation
    pf1 = probabilistic_f1(all_targets.flatten(), all_probs.flatten())

    return val_loss, pf1


def run_training(debug=Config.DEBUG):
    """
    Main function to run the training pipeline.
    """
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Determine tabular input dimension from a batch
    # We grab one batch from train_loader to inspect shapes
    temp_images, temp_tab, temp_targets = next(iter(train_loader))
    tabular_input_dim = temp_tab.shape[1]
    print(f"Tabular Input Dimension: {tabular_input_dim}")

    # 2. Initialize Model
    print("Initializing model...")
    model = HybridEfficientNet(tabular_input_dim=tabular_input_dim)
    model = model.to(device)

    # 3. Define Loss, Optimizer, Scheduler
    # Handle class imbalance with pos_weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    # pct_start determines the percentage of the cycle spent increasing the LR
    pct_start = Config.WARMUP_EPOCHS / Config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        div_factor=25.0,  # Initial LR = max_lr / 25
        final_div_factor=100.0,  # Final LR = Initial LR / 100
    )

    # 4. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_fn(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_pf1 = eval_fn(model, val_loader, criterion, device)

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val pF1: {val_pf1}")

        # Checkpointing and Early Stopping
        if val_pf1 > best_pf1:
            print(
                f"Validation pF1 improved from {best_pf1} to {val_pf1}. Saving model..."
            )
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val pF1: {best_pf1}")
