import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import prepare_data, GIDataset, get_transforms
from library.model import UNetPlusPlus25D
from library.losses import BCETverskyLoss
from library.inference import predict_sliding_window
from library.utils import calculate_dice


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, data in enumerate(loader):
        images = data["image"].to(device)
        masks = data["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # With Deep Supervision, outputs is a list of tensors
        outputs = model(images)

        # Compute loss
        loss = loss_fn(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Validates the model using sliding window inference on full images.
    Returns average Dice score.
    """
    model.eval()
    dice_scores = []

    # We use no_grad context, but predict_sliding_window also uses it internally.
    # We iterate over the validation set. Batch size is expected to be 1.
    with torch.no_grad():
        for data in loader:
            # data['image'] is (B, C, H, W). B=1
            # predict_sliding_window expects (C, H, W)
            image_tensor = data["image"][0]
            mask_numpy = data["mask"][0].numpy()  # (C, H, W)

            # Predict using sliding window to handle full resolution
            # Returns numpy array (C, H, W) with probabilities
            pred_probs = predict_sliding_window(model, image_tensor, device)

            # Threshold predictions
            pred_mask = (pred_probs > 0.5).astype(np.uint8)

            # Calculate Dice for this slice (average over classes)
            # mask_numpy is (C, H, W), pred_mask is (C, H, W)
            # We flatten or calculate per class. calculate_dice expects flat or same shape.
            # The metric is usually mean dice over classes.

            slice_dice = 0.0
            for c in range(Config.NUM_CLASSES):
                d = calculate_dice(mask_numpy[c], pred_mask[c])
                slice_dice += d

            dice_scores.append(slice_dice / Config.NUM_CLASSES)

    return np.mean(dice_scores)


def run_training():
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)

    print(f"Initializing training on device: {Config.DEVICE}")

    # 1. Prepare Data
    # Load metadata
    df_train = prepare_data(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    df_val = prepare_data(Config.VAL_METADATA_PATH, mode="val", load_cached_data=True)

    # Debug mode: subset data
    if Config.DEBUG:
        print("DEBUG Mode: Using subset of data.")
        df_train = df_train.head(100)
        df_val = df_val.head(20)

    # Create Datasets
    train_dataset = GIDataset(
        df_train, mode="train", transforms=get_transforms("train")
    )
    val_dataset = GIDataset(df_val, mode="val", transforms=get_transforms("val"))

    # Create DataLoaders
    # Train loader: shuffle=True, drop_last=True to avoid batch norm issues with small last batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val loader: batch_size=1 for sliding window inference
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Initialize Model, Loss, Optimizer
    model = UNetPlusPlus25D().to(Config.DEVICE)
    loss_fn = BCETverskyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # 3. Training Loop
    best_dice = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, Config.DEVICE
        )

        # Validate
        val_dice = validate(model, val_loader, Config.DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {epoch_duration:.0f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Dice: {val_dice:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved ({best_dice:.6f} -> {val_dice:.6f}). Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(
        f"Training finished in {total_time/60:.1f} minutes. Best Val Dice: {best_dice:.6f}"
    )


if __name__ == "__main__":
    run_training()
