import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import config
from library.model import ECM_MLP
from library.dataset import get_dataloaders


def train_model():
    """
    Executes the training pipeline for the ECM_MLP model.

    This function:
    1. Sets up the device (GPU/CPU).
    2. Loads data using the centralized data loader.
    3. Initializes the model, loss function (MAE), optimizer (AdamW), and scheduler.
    4. Runs the training loop with validation.
    5. Implements early stopping and saves the best model checkpoint.

    Returns:
        float: The best validation loss achieved.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_dataloaders handles caching and preprocessing internally via library.preprocessing
    train_loader, val_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=4
    )

    # 3. Model Initialization
    model = ECM_MLP().to(device)

    # 4. Optimization Setup
    # L1 Loss is used as it corresponds to Mean Absolute Error (MAE)
    criterion = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Reduce learning rate when a metric has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * features.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * features.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # --- Reporting ---
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"Train Loss: {epoch_train_loss}")
        print(f"Val Loss: {epoch_val_loss}")

        # --- Scheduling & Checkpointing ---
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        # --- Early Stopping ---
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return best_val_loss
