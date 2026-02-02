import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time

from library.config import Config
from library.dataset import get_loaders
from library.model import BiSeNet25D
from library.loss import BCEDiceLoss
from library.utils import dice_coef


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass: BiSeNet returns (main_out, aux_out)
        outputs = model(images)

        # Loss calculation (BCEDiceLoss handles the tuple unpacking and weighting)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass: We only care about main output for validation
            main_out, _ = model(images)

            # Calculate loss just for monitoring (using main output vs target)
            loss = criterion(main_out, masks)
            running_loss += loss.item()

            # Calculate Dice Metric
            # Apply sigmoid to logits
            preds_prob = torch.sigmoid(main_out)

            # Convert to numpy for metric calculation
            preds_np = preds_prob.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Threshold at 0.5
            preds_bin = (preds_np > 0.5).astype(np.float32)

            # Compute Dice
            batch_dice = dice_coef(masks_np, preds_bin)
            dice_scores.append(batch_dice)

    epoch_loss = running_loss / len(loader)
    epoch_dice = np.mean(dice_scores)

    return epoch_loss, epoch_dice


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    patience=5,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline with Early Stopping.
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata is generated."
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debugging: Reduce dataset size if configured
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_loaders(
        train_df,
        val_df,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    print(f"Initializing {Config.MODEL_NAME}...")
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = BCEDiceLoss(bce_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5. Training Loop
    best_dice = 0.0
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Val Dice: {val_dice:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(
        f"Training complete. Total time: {total_time:.2f}s. Best Val Dice: {best_dice:.10f}"
    )
