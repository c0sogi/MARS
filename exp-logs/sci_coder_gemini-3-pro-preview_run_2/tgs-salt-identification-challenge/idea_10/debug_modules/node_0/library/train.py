import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    SEED,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_T_MAX,
    CHECKPOINT_PATH,
    WORKING_DIR,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.dataset import SaltDataset
from library.model import DepthRobustLinkNet
from library.losses import CombinedLoss
from library.utils import do_kaggle_metric, save_checkpoint


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks, depths, _) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, depths)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Performs a linear search for the optimal binarization threshold.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            loss = criterion(logits, masks)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            # Store predictions and masks for threshold search
            # Move to CPU to save GPU memory
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Linear search for optimal threshold
    # Search range: 0.3 to 0.7 with step 0.05
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        score = do_kaggle_metric(all_preds, all_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return avg_loss, best_score, best_threshold


def train_model():
    """
    Main function to orchestrate the training process.
    """
    set_seed(SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Initialize Datasets and Loaders
    train_dataset = SaltDataset(mode="train", load_cached_data=True)
    val_dataset = SaltDataset(mode="val", load_cached_data=True)

    if DEBUG:
        # Subset for debugging
        indices = list(range(min(len(train_dataset), DEBUG_SAMPLE_SIZE)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model, Loss, Optimizer
    model = DepthRobustLinkNet(in_channels=1, n_classes=1)
    model = model.to(DEVICE)

    criterion = CombinedLoss(bce_weight=0.5, lovasz_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=1e-6
    )

    # 3. Training Loop
    best_map = 0.0
    patience = 10
    patience_counter = 0

    print("Starting training...")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_map, best_thresh = validate(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{EPOCHS} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val mAP: {val_map}")
        print(f"Best Threshold: {best_thresh}")

        # Checkpointing and Early Stopping
        if val_map > best_map:
            print(
                f"Validation mAP improved from {best_map} to {val_map}. Saving checkpoint..."
            )
            best_map = val_map
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_map, CHECKPOINT_PATH)

            # Save the best threshold to a text file for inference reference
            with open(os.path.join(WORKING_DIR, "best_threshold.txt"), "w") as f:
                f.write(str(best_thresh))
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        print("-" * 30)

    print("Training complete.")
    print(f"Best Validation mAP: {best_map}")
