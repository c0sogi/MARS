import torch
import torch.optim as optim
import numpy as np
import os
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import ResidualFCN
from library.losses import BCEDiceLoss
from library.utils import ModelEMA, fbeta_score


def train_one_epoch(model, loader, optimizer, loss_fn, device, ema=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the F0.5 score.
    Uses the deterministic grid provided by the dataloader.
    """
    model.eval()
    running_score = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            # Unpack based on what the dataset returns for validation
            # InkDataset val returns: vol_tensor, label_tensor, coords
            images, labels, _ = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Calculate F0.5 score for this batch
            # We use the threshold defined in Config
            score = fbeta_score(outputs, labels, beta=0.5, threshold=Config.THRESHOLD)

            running_score += score * images.size(0)
            count += images.size(0)

    return running_score / count if count > 0 else 0.0


def train(config=Config):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    # Ensure output directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # DataLoaders
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if not train_loader:
        print("Error: No training data found.")
        return

    # Model
    model = ResidualFCN().to(device)

    # EMA
    ema = ModelEMA(model, decay=config.EMA_DECAY)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = BCEDiceLoss(bce_weight=config.BCE_WEIGHT)

    # Training State
    best_score = -1.0
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(
        f"Config: Epochs={config.EPOCHS}, Batch={config.BATCH_SIZE}, LR={config.LEARNING_RATE}"
    )

    for epoch in range(config.EPOCHS):
        # --- Training ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, ema
        )

        # --- Validation ---
        # We evaluate the EMA model as it is expected to be more robust
        val_score = 0.0
        if val_loader:
            val_score = validate(ema.get_model(), val_loader, device)

        # --- Logging ---
        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss} - Val F0.5: {val_score}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_loader:
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                # Save the best EMA model
                torch.save(ema.get_model().state_dict(), config.MODEL_PATH)
                print(f"New best model saved to {config.MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        else:
            # If no validation set, just save the latest model
            torch.save(ema.get_model().state_dict(), config.MODEL_PATH)

    print(f"Training finished. Best Val F0.5: {best_score}")
