import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.model import DS_AG_CAC_ResUNet
from library.dataset import DenoisingDataset
from library.loss import MultiScaleMSELoss
from library.utils import seed_everything, get_device, calculate_rmse
from library.inference import predict_tiled, generate_submission


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy, clean, _) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns [final_output, aux_output1, aux_output2, ...]
        preds = model(noisy)

        # Compute loss
        loss = criterion(preds, noisy, clean)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using tiled inference.
    """
    model.eval()
    total_rmse = 0.0

    with torch.no_grad():
        for i, (noisy, clean, _) in enumerate(loader):
            noisy = noisy.to(device)
            # clean is (1, 1, H, W) ground truth

            # predict_tiled returns the predicted CLEAN image (1, C, H, W)
            # It handles TTA and sliding window internally based on Config
            pred_clean = predict_tiled(model, noisy, device)

            # Calculate RMSE between predicted clean and ground truth clean
            rmse = calculate_rmse(clean, pred_clean)
            total_rmse += rmse

    return total_rmse / len(loader)


def train_model():
    """
    Main function to manage the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Using load_cached_data=True as per requirements to utilize caching logic
    train_dataset = DenoisingDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    val_dataset = DenoisingDataset(
        Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation batch size is 1 to handle full images
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = DS_AG_CAC_ResUNet().to(device)

    # 4. Optimization
    # Define weights for Deep Supervision (decaying weights for aux heads)
    # Outputs: [Final, Aux4, Aux3, Aux2]
    loss_weights = [1.0, 0.5, 0.5, 0.5] if Config.USE_DEEP_SUPERVISION else [1.0]
    criterion = MultiScaleMSELoss(weights=loss_weights).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LEARNING_RATE
    )

    # 5. Training Loop
    best_rmse = float("inf")
    patience = 10  # Early stopping patience
    patience_counter = 0

    print(f"Starting Training for {Config.NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_rmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Print metrics (Full precision for Val RMSE)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val RMSE: {val_rmse}"
        )

        # Checkpoint & Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  -> New Best Model Saved! RMSE: {best_rmse}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Val RMSE: {best_rmse}")

    # 6. Generate Submission
    # Uses the best model saved at Config.MODEL_CHECKPOINT_PATH
    print("Generating Submission...")
    generate_submission(Config.MODEL_CHECKPOINT_PATH, Config.SUBMISSION_PATH)
