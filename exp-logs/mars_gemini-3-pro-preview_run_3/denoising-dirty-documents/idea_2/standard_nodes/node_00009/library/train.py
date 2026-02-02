import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config, seed_everything
from library.model import UNet, train_one_epoch, validate
from library.dataset import get_dataloaders
from library.utils import save_checkpoint, load_checkpoint, generate_submission_file


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug_sample_size (int, optional): Number of samples to use for debugging.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        debug_sample_size=debug_sample_size,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS).to(
        device
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step(val_rmse)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val RMSE = {val_rmse:.10f}"
        )

        # Checkpointing and Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_rmse, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Best Validation RMSE: {best_rmse:.10f}")

    # 5. Inference on Test Set
    print("Generating submission for test set...")

    # Load best model weights
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    if checkpoint is None:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()
    predictions = {}

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            # inputs shape: [1, C, H, W]

            # Pad input to be divisible by 16 (requirement for U-Net pooling)
            h, w = inputs.shape[2], inputs.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                inputs_padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                inputs_padded = inputs

            # Forward pass
            outputs_padded = model(inputs_padded)

            # Crop back to original dimensions
            outputs = outputs_padded[:, :, :h, :w]

            # Clamp pixel values to valid range [0, 1]
            outputs = torch.clamp(outputs, 0, 1)

            # Convert to numpy array (H, W)
            pred_img = outputs.squeeze().cpu().numpy()

            # Store prediction
            img_id = test_ids[i]
            predictions[img_id] = pred_img

    # 6. Save Submission
    generate_submission_file(predictions, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
