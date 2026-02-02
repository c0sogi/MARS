import os
import torch
import torch.nn as nn
from library import config, data, model as lib_model


def run_training(num_epochs=config.NUM_EPOCHS, load_cached_data=True, patience=5):
    """
    Executes the training pipeline for the Asymmetric EfficientNet model.

    This function manages the training loop, validation, metric logging,
    checkpointing of the best model, and early stopping.

    Args:
        num_epochs (int): Maximum number of training epochs.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
        patience (int): Number of epochs to wait for improvement in Validation AUC before stopping.

    Returns:
        float: The best validation AUC score achieved.
    """
    # 1. Setup Environment
    config.seed_everything(config.SEED)
    device = config.DEVICE

    # Ensure working directory exists for checkpoints
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")

    # 2. Load Data
    # Uses the library function which handles caching logic internally
    train_loader, val_loader, _ = data.get_data_loaders(
        load_cached_data=load_cached_data
    )

    # 3. Initialize Model, Criterion, and Optimizer
    model = lib_model.AsymmetricEfficientNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Execute one training epoch
        train_loss, train_auc = lib_model.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Execute validation
        val_loss, val_auc = lib_model.validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"No improvement in Validation AUC. Patience: {epochs_no_improve}/{patience}"
            )

        # Early Stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    return best_val_auc
