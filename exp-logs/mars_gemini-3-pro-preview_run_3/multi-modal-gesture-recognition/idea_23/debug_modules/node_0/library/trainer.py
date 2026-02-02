import os
import torch
import numpy as np
from library import config, model, loss

# ==========================================
# Reproducibility
# ==========================================
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


def train_epoch(model_instance, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model_instance.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (features, targets, _, _) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model_instance(features)

        # Compute loss
        loss_val = criterion(outputs, targets)

        # Backward pass
        loss_val.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss_val.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(model_instance, loader, criterion, device):
    """
    Performs validation on the validation set.
    Returns average loss and frame-wise accuracy of the final stage.
    """
    model_instance.eval()
    running_loss = 0.0
    correct_frames = 0
    total_frames = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (features, targets, _, _) in enumerate(loader):
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model_instance(features)

            # Compute loss
            loss_val = criterion(outputs, targets)
            running_loss += loss_val.item()
            num_batches += 1

            # Compute Accuracy on Stage 3 (Final Output)
            # outputs['logits_3'] shape: (Batch, Time, Classes)
            logits = outputs["logits_3"]
            predictions = torch.argmax(logits, dim=2)  # (Batch, Time)

            # Flatten for comparison
            preds_flat = predictions.view(-1)
            targets_flat = targets.view(-1)

            correct_frames += (preds_flat == targets_flat).sum().item()
            total_frames += targets_flat.size(0)

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct_frames / total_frames if total_frames > 0 else 0.0

    return avg_loss, accuracy


def train_model(train_loader, val_loader):
    """
    Main function to train the WES-KN model.
    Handles initialization, training loop, early stopping, and saving.
    """
    device = torch.device(config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Initialize Model, Loss, Optimizer
    weskn_model = model.WESKN().to(device)
    criterion = loss.CascadedSmoothnessLoss()

    # Using standard Adam as per prompt requirements (Idea 23)
    optimizer = torch.optim.Adam(
        weskn_model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # 2. Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(
            weskn_model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(weskn_model, val_loader, criterion, device)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val Accuracy: {val_acc}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(weskn_model.state_dict(), config.MODEL_SAVE_PATH)
            print("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return weskn_model
