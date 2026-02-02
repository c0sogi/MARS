import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, set_seed
from library.data import get_dataloaders
from library.model import AttentivePyramidSiamese
from library.utils import probabilistic_f1


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The computing device (CPU/GPU).

    Returns:
        avg_loss (float): Average loss over the epoch.
        train_pf1 (float): Probabilistic F1 score on the training set.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch_idx, (target_input, contra_input, labels) in enumerate(loader):
        # Move data to device
        target_input = target_input.to(device)
        contra_input = contra_input.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (B, 1)

        # Forward pass
        # Mixed precision is enabled in Config, so we should ideally use autocast
        # However, to keep it simple and robust as per instructions without explicit scaler provided in library,
        # we will use standard float32 or rely on PyTorch's automatic handling if configured globally.
        # Given the prompt doesn't provide a GradScaler in utils, we proceed with standard execution
        # or simple autocast context if Config.USE_AMP is True.

        if Config.USE_AMP and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(target_input, contra_input)
                loss = criterion(logits, labels)
        else:
            logits = model(target_input, contra_input)
            loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping is EXPLICITLY DISABLED per instructions

        optimizer.step()

        # Statistics
        running_loss += loss.item() * target_input.size(0)

        # Store predictions for pF1 calculation
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_preds.append(probs)
        all_targets.append(targets)

    # Aggregate metrics
    dataset_size = len(loader.dataset)
    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    train_pf1 = probabilistic_f1(all_targets, all_preds)

    return avg_loss, train_pf1


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The computing device.

    Returns:
        avg_loss (float): Average validation loss.
        val_pf1 (float): Probabilistic F1 score on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for target_input, contra_input, labels in loader:
            target_input = target_input.to(device)
            contra_input = contra_input.to(device)
            labels = labels.to(device).unsqueeze(1)

            if Config.USE_AMP and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(target_input, contra_input)
                    loss = criterion(logits, labels)
            else:
                logits = model(target_input, contra_input)
                loss = criterion(logits, labels)

            running_loss += loss.item() * target_input.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    dataset_size = len(loader.dataset)
    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_pf1 = probabilistic_f1(all_targets, all_preds)

    return avg_loss, val_pf1


def run_training(load_cached_data=True):
    """
    Main execution function to train the Attentive Pyramid Symmetry-Difference Siamese Network.

    Args:
        load_cached_data (bool): Whether to load pre-computed data/stats from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    print("Initializing Model...")
    model = AttentivePyramidSiamese(
        backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED
    )
    model.to(device)

    # 4. Loss Function
    # Weighted BCE to handle 1:47 imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 6. Training Loop
    best_val_pf1 = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss} | Train pF1: {train_pf1}")
        print(f"  Val Loss:   {val_loss} | Val pF1:   {val_pf1}")

        # Checkpointing & Early Stopping
        if val_pf1 > best_val_pf1:
            print(
                f"  [Improved] val_pf1 increased from {best_val_pf1} to {val_pf1}. Saving model..."
            )
            best_val_pf1 = val_pf1
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation pF1: {best_val_pf1}")
    return best_val_pf1
