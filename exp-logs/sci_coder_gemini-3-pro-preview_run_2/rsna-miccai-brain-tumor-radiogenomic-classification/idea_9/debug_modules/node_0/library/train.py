import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from library.config import Config
from library.data import get_dataloaders
from library.model import (
    AsymmetricEfficientNet,
    train_one_epoch,
    validate,
    generate_submission,
)


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
):
    """
    Executes the training pipeline.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for dataloaders.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        patience (int): Early stopping patience.
    """
    # Update Config with runtime arguments to ensure consistency across modules
    Config.NUM_EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate
    Config.WEIGHT_DECAY = weight_decay
    Config.PATIENCE = patience

    # Set reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Loading
    # The caching logic is handled inside get_dataloaders via prepare_data_split
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 3. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler to reduce LR when validation AUC plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        # Train Step
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Log metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f}, Train AUC: {train_auc:.10f}, "
            f"Val Loss: {val_loss:.10f}, Val AUC: {val_auc:.10f}"
        )

        # Update Scheduler
        scheduler.step(val_auc)

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # 5. Inference
    # Load the best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Generate submission using TTA
    generate_submission(model, test_loader, test_ids, device)
