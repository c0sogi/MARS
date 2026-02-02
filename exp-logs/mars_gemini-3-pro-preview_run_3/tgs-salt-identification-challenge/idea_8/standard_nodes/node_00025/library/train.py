import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calc_map
from library.dataset import SaltDataset, get_transforms, load_data
from library.model import SaltModel
from library.losses import BCEDiceLoss, LovaszHingeLoss


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    """
    Handles the training of one epoch.
    Implements Deep Supervision loss aggregation and Mixed Precision.
    """
    model.train()
    running_loss = 0.0

    # Determine loss mode based on curriculum
    use_lovasz = epoch >= Config.LOVASZ_SWITCH_EPOCH

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            # Model returns a list of outputs in train mode (Deep Supervision)
            outputs = model(images)

            loss = 0
            if use_lovasz:
                # Fine-tuning: Optimize only the final head with Lovasz
                # outputs[-1] is the final prediction
                loss = criterion(outputs[-1], masks)
            else:
                # Warm-up: Optimize all heads with BCE+Dice
                # Deep supervision: sum loss from all heads
                for output in outputs:
                    loss += criterion(output, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device, epoch):
    """
    Evaluates the model on the validation set.
    Computes Loss and Mean Average Precision (mAP).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    use_lovasz = epoch >= Config.LOVASZ_SWITCH_EPOCH

    # Accumulate predictions for threshold optimization (Cite Lesson 00011)
    all_preds_np = []
    all_masks_np = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Model returns single tensor in eval mode
            output = model(images)

            # Calculate loss for monitoring
            loss = criterion(output, masks)
            running_loss += loss.item()

            # Apply sigmoid and move to CPU
            preds = torch.sigmoid(output).cpu().numpy()
            masks_np = masks.cpu().numpy()

            all_preds_np.append(preds)
            all_masks_np.append(masks_np)

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds_np = np.concatenate(all_preds_np, axis=0)
    all_masks_np = np.concatenate(all_masks_np, axis=0)

    # Dynamic Threshold Sweep (Cite Lesson 00011)
    # Optimize metric to decouple discrimination from calibration
    best_map = 0.0
    thresholds = np.arange(0.3, 0.76, 0.05)

    for t in thresholds:
        score = calc_map(all_preds_np, all_masks_np, pixel_threshold=t)
        if score > best_map:
            best_map = score

    return avg_loss, best_map


def run_fold(fold_idx, debug=False):
    """
    Orchestrates the training pipeline for a single fold.
    """
    seed_everything(Config.SEED)

    print(f"\n{'='*20} Starting Fold {fold_idx+1}/{Config.N_FOLDS} {'='*20}")

    # 1. Load Metadata
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if debug:
        df = df.head(100)  # Reduced dataset for debugging
        print("DEBUG MODE: Using 100 samples.")

    # 2. Load Data (Images/Masks) into Memory
    # This handles caching automatically
    ids, images, masks, depths = load_data(
        df, cache_name="train", load_cached_data=True
    )

    # 3. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We split based on coverage_class to ensure balanced salt amounts
    # df has 'coverage_class' from metadata generation
    fold_generator = skf.split(ids, df["coverage_class"])

    # Get indices for the specific fold
    train_idx, val_idx = list(fold_generator)[fold_idx]

    # Slice data
    X_train, d_train, y_train = images[train_idx], depths[train_idx], masks[train_idx]
    X_val, d_val, y_val = images[val_idx], depths[val_idx], masks[val_idx]

    print(f"Train set: {len(X_train)} samples")
    print(f"Val set:   {len(X_val)} samples")

    # 4. Datasets and Loaders
    train_dataset = SaltDataset(
        X_train, d_train, y_train, transform=get_transforms("train")
    )
    val_dataset = SaltDataset(X_val, d_val, y_val, transform=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Setup
    device = Config.DEVICE
    model = SaltModel(
        encoder_name=Config.ENCODER, pretrained=True, in_channels=Config.CHANNELS
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when mAP plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    scaler = GradScaler()

    # Losses
    criterion_bce = BCEDiceLoss()
    criterion_lovasz = LovaszHingeLoss()

    # Tracking
    best_map = 0.0
    patience_counter = 0

    # 6. Training Loop
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Select Criterion based on Epoch
        if epoch < Config.LOVASZ_SWITCH_EPOCH:
            criterion = criterion_bce
            loss_name = "BCE+Dice"
        else:
            criterion = criterion_lovasz
            loss_name = "Lovasz"

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch
        )

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, device, epoch)

        # Scheduler Step
        scheduler.step(val_map)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} [{loss_name}] - "
            f"Time: {elapsed:.1f}s - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val mAP: {val_map}"
        )  # Full precision printing

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best mAP! Model saved to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

        # Ensure we don't stop right before the switch if patience is tight,
        # though Config says patience 15 and switch 15, so it should be fine.

    print(f"Fold {fold_idx+1} finished. Best mAP: {best_map}")

    # Clear memory
    del model, optimizer, scaler, train_loader, val_loader, X_train, X_val
    torch.cuda.empty_cache()

    return best_map
