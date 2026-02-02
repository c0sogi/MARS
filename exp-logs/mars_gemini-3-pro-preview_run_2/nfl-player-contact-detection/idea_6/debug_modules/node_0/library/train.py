import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import FocalLoss, optimize_threshold, calc_mcc
from library.model import CKResNet
from library.dataset import get_dataloaders


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The CKResNet model.
        loader: Training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: Torch device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        # Unpack inputs: (x_wide, x_center, condition)
        x_wide, x_center, condition = inputs

        # Move to device
        x_wide = x_wide.to(device)
        x_center = x_center.to(device)
        condition = condition.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x_wide, x_center, condition)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The CKResNet model.
        loader: Validation DataLoader.
        device: Torch device.

    Returns:
        tuple: (y_probs, y_true) as numpy arrays.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            x_wide, x_center, condition = inputs

            x_wide = x_wide.to(device)
            x_center = x_center.to(device)
            condition = condition.to(device)

            # Forward pass
            logits = model(x_wide, x_center, condition)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.numpy())

    y_probs = np.concatenate(preds_list).flatten()
    y_true = np.concatenate(targets_list).flatten()

    return y_probs, y_true


def run_training(debug=False, load_cached_data=True):
    """
    Main function to orchestrate the training process.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, tries to load pre-processed features.

    Returns:
        tuple: (best_model, best_threshold)
    """
    # 1. Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Config.setup_directories()

    # 2. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Initialize Model
    # Determine input dimensions from a sample batch
    sample_inputs, _ = next(iter(train_loader))
    input_dim = sample_inputs[0].shape[1]
    center_dim = sample_inputs[1].shape[1]

    print(f"Model Input Dim: {input_dim}, Center Dim: {center_dim}")

    model = CKResNet(input_dim=input_dim, center_dim=center_dim).to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss()

    # 5. Training Loop
    best_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_probs, val_true = validate(model, val_loader, device)

        # Optimize Threshold & Calculate MCC
        curr_threshold, curr_mcc = optimize_threshold(val_true, val_probs)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Validation MCC: {curr_mcc}")
        print(f"Best Threshold: {curr_threshold}")

        # Early Stopping & Checkpointing
        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_threshold
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation MCC: {best_mcc}")
    print(f"Best Threshold: {best_threshold}")

    # Load best weights before returning
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH))

    return model, best_threshold
