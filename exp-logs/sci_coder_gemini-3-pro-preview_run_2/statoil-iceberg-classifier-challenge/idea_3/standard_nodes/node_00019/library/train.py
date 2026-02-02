import torch
import torch.nn as nn
import torch.optim as optim
import copy
from library.config import Config
from library.model import SAHCN
from library.data_loader import get_dataloaders
from library.utils import save_checkpoint, print_metrics


def train_model():
    """
    Executes the training pipeline for the SAHCN model.

    Steps:
    1. Initializes model, optimizer, loss function, and scheduler.
    2. Runs the training loop with validation monitoring.
    3. Implements Early Stopping based on validation loss.
    4. Saves the best model weights.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    model = SAHCN().to(device)

    # Get DataLoaders (handles caching internally)
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Define Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.LR_MIN,
    )

    # Early Stopping Variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # 2. Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for inputs, meta, labels in train_loader:
            inputs = inputs.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, meta)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        corrects = 0

        with torch.no_grad():
            for inputs, meta, labels in val_loader:
                inputs = inputs.to(device)
                meta = meta.to(device)
                labels = labels.to(device)

                outputs = model(inputs, meta)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                corrects += torch.sum(preds == labels).item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects / len(val_loader.dataset)

        # --- Scheduler Step ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)

        # --- Logging ---
        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": epoch_train_loss,
            "Val Loss": epoch_val_loss,
            "Val Acc": epoch_val_acc,
            "LR": current_lr,
        }
        print_metrics(metrics)

        # --- Early Stopping Logic ---
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered. No improvement for {Config.PATIENCE} epochs."
            )
            break

    # 3. Finalization
    print(f"Training complete. Best Validation Loss: {best_loss}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save to disk
    save_checkpoint(model, Config.MODEL_SAVE_PATH)
    print(f"Best model saved to {Config.MODEL_SAVE_PATH}")

    return model
