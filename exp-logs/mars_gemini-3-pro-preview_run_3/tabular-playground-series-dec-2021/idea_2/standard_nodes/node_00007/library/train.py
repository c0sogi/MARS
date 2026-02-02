import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import ResNetMLP, train_one_epoch, validate, predict_and_submit


def run_training(
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 5,
    debug_limit: int = None,
    num_workers: int = 4,
):
    """
    Orchestrates the training process for the Forest Cover Type prediction.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        lr (float): Learning rate for the optimizer.
        patience (int): Early stopping patience epochs.
        debug_limit (int, optional): Limit dataset size for debugging.
        num_workers (int): Number of workers for data loading.
    """

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_everything(42)

    # 2. Data Loading
    # Input features: 13 numerical (3 new) + 44 binary = 57
    # Classes: 6 (mapped from original 7 classes, excluding class 5)
    input_dim = 57
    num_classes = 6

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=True,
        debug_limit=debug_limit,
    )

    # 3. Model Initialization
    model = ResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        num_blocks=3,
        hidden_dim=256,
        dropout_rate=0.2,
    )
    model = model.to(device)

    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision as required
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 6. Load Best Model
    if best_model_state is not None:
        print("Loading best model weights for inference...")
        model.load_state_dict(best_model_state)

    # 7. Prediction & Submission
    output_path = "./submission/submission.csv"
    predict_and_submit(model, test_loader, output_path=output_path, device=device)
