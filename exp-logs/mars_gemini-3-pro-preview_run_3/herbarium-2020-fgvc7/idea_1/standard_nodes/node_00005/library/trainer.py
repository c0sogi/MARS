import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import library.config as config
import library.dataset as dataset
import library.model as model_lib


def run_training(
    load_cached_data=True,
    max_train_samples=None,
    max_val_samples=None,
    epochs=config.NUM_EPOCHS,
    patience=config.PATIENCE,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
):
    """
    Manages the training pipeline: data loading, model initialization,
    training loop with early stopping, and submission generation.

    Args:
        load_cached_data (bool): Whether to load cached dataset artifacts.
        max_train_samples (int, optional): Limit training data for debugging.
        max_val_samples (int, optional): Limit validation data for debugging.
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        batch_size (int): Batch size for dataloaders.
        learning_rate (float): Learning rate for the optimizer.
    """

    # 1. Setup Configuration
    # Set seed for reproducibility
    config.set_seed(config.SEED)

    # Update global config based on function arguments
    # This ensures dataset and model libraries use the correct runtime parameters
    config.MAX_TRAIN_SAMPLES = max_train_samples
    config.MAX_VAL_SAMPLES = max_val_samples
    config.NUM_EPOCHS = epochs
    config.PATIENCE = patience
    config.BATCH_SIZE = batch_size
    config.LEARNING_RATE = learning_rate

    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # get_dataloaders handles caching internally via _get_train_weights
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print("Initializing model...")
    net = model_lib.ResNet18Classifier(num_classes=config.NUM_CLASSES, pretrained=True)
    net = net.to(device)

    # 4. Optimizer & Criterion Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    # 5. Training Loop
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = float("inf")
    best_f1 = 0.0
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        # Train Phase
        train_loss, train_acc = model_lib.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")

        # Validation Phase
        val_loss, val_acc, val_f1 = model_lib.validate(
            net, val_loader, criterion, device
        )
        # Printing metrics with full precision (no formatting) as required
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")
        print(f"Val F1: {val_f1}")

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping & Checkpointing Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(net.state_dict())

            # Save the best model checkpoint
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved. New best Val Loss: {best_loss}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {best_loss}")
    print(f"Best Val F1: {best_f1}")

    # Submission generation is now handled by runfile.py based on metric check
