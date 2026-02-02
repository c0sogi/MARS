import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import set_seed, get_device, save_checkpoint
from library.data import get_dataloaders
from library.model import (
    AsymmetricEfficientNet,
    train_one_epoch,
    validate,
    generate_submission,
)


def run_training(
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    patience=5,
    debug_limit=None,
    load_cached_data=True,
):
    """
    Orchestrates the training process, including data loading, model initialization,
    training loop, validation, early stopping, and submission generation.
    """
    # 1. Setup
    set_seed(42)
    device = get_device()

    # 2. Data Loading
    # Uses the caching mechanism implemented in library.data via load_cached_data=True
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Model Initialization
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.5)
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = "./working/idea_10_float32/best_model.pth"

    # Ensure working directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        # Train & Validate
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train AUC: {train_auc} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Scheduler Step
        scheduler.step(val_auc)

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Final Inference
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate submission using Test-Time Augmentation (implemented in library.model)
    generate_submission(
        model, test_loader, device, output_path="./submission/submission.csv"
    )
