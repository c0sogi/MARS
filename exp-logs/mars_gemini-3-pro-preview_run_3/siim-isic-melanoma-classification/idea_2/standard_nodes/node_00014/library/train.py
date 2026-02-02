import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    POS_WEIGHT,
    PATIENCE,
    SUBMISSION_PATH,
    SEED,
    WORKING_DIR,
)
from library.utils import (
    seed_everything,
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import HybridEfficientNet


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()

    losses = AverageMeter()
    # We will store targets and preds to calculate AUC for the epoch
    all_targets = []
    all_preds = []

    for batch_idx, data in enumerate(train_loader):
        images = data["image"].to(device)
        meta = data["meta"].to(device)
        targets = data["target"].to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        logits = model(images, meta)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        # Store predictions for AUC calculation
        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets_np)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, epoch_auc


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set using 4-view TTA.
    Cite solution_lesson_node_00012.
    """
    model.eval()

    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images = data["image"].to(device)
            meta = data["meta"].to(device)
            targets = data["target"].to(device).unsqueeze(1)

            # Original view for loss calculation
            logits = model(images, meta)
            loss = criterion(logits, targets)
            losses.update(loss.item(), images.size(0))

            # TTA: Original + HFlip + VFlip + HVFlip (180)
            logits1 = logits
            logits2 = model(torch.flip(images, [2]), meta)  # Vertical
            logits3 = model(torch.flip(images, [3]), meta)  # Horizontal
            logits4 = model(torch.flip(images, [2, 3]), meta)  # Both

            # Average probabilities
            probs = (
                torch.sigmoid(logits1)
                + torch.sigmoid(logits2)
                + torch.sigmoid(logits3)
                + torch.sigmoid(logits4)
            ) / 4.0

            preds = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets_np)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    val_auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, val_auc


def inference(test_loader, model, device):
    """
    Generates predictions for the test set using 4-view TTA.
    Cite solution_lesson_node_00012.
    """
    model.eval()

    image_names = []
    predictions = []

    print("Starting inference on test set with TTA...")

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images = data["image"].to(device)
            meta = data["meta"].to(device)
            names = data["image_name"]

            # TTA: Original + HFlip + VFlip + HVFlip
            logits1 = model(images, meta)
            logits2 = model(torch.flip(images, [2]), meta)
            logits3 = model(torch.flip(images, [3]), meta)
            logits4 = model(torch.flip(images, [2, 3]), meta)

            probs = (
                torch.sigmoid(logits1)
                + torch.sigmoid(logits2)
                + torch.sigmoid(logits3)
                + torch.sigmoid(logits4)
            ) / 4.0

            probs = probs.cpu().numpy().flatten()

            image_names.extend(names)
            predictions.extend(probs)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"image_name": image_names, "target": predictions})

    # Save to CSV
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")

    return submission_df


def run_training():
    """
    Main driver function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(SEED)

    # 2. Data Loading
    print("Loading data...")
    # Using cached data if available as per requirements
    train_loader, val_loader, test_loader, meta_dim = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print(f"Initializing HybridEfficientNet with meta_dim={meta_dim}...")
    model = HybridEfficientNet(meta_dim=meta_dim)
    model.to(DEVICE)

    # 4. Loss, Optimizer, Scheduler
    # Dampened positive weight for class imbalance
    pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_score = 0.0
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            train_loader, model, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_auc = validate(val_loader, model, criterion, DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{EPOCHS} [{elapsed:.0f}s]: "
            f"LR={current_lr:.6f}, "
            f"Train Loss={train_loss}, Train AUC={train_auc}, "
            f"Val Loss={val_loss}, Val AUC={val_auc}"
        )

        # Checkpointing
        is_best = val_auc > best_score
        if is_best:
            best_score = val_auc
            patience_counter = 0
            print(f"New best score: {best_score}")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
            },
            is_best,
        )

        # Early Stopping
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    # 6. Inference
    print("Loading best model for inference...")
    best_score_loaded = load_checkpoint(model, filename="best_model.pth")
    print(f"Loaded model with validation AUC: {best_score_loaded}")

    inference(test_loader, model, DEVICE)
