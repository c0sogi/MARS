import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, weighted_auc, seed_everything
from library.model import ResV2GeM


def train_one_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()

        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, targets)

            # Sigmoid for probability
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))

            # Store for metric calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
    else:
        all_targets = np.array([])
        all_preds = np.array([])

    # Calculate Weighted AUC
    score = weighted_auc(all_targets, all_preds)

    return score, losses.avg


def predict_tta(model, loader, device):
    """
    Generates predictions using 5-view Test Time Augmentation.
    Views: Original, HFlip, VFlip, Rot90, Rot270.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for image_ids, images in loader:
            images = images.to(device, non_blocking=True)

            # 1. Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. Horizontal Flip
            images_h = torch.flip(images, [3])
            logits_h = model(images_h)
            probs += torch.sigmoid(logits_h)

            # 3. Vertical Flip
            images_v = torch.flip(images, [2])
            logits_v = model(images_v)
            probs += torch.sigmoid(logits_v)

            # 4. Rot90
            images_r90 = torch.rot90(images, 1, [2, 3])
            logits_r90 = model(images_r90)
            probs += torch.sigmoid(logits_r90)

            # 5. Rot270
            images_r270 = torch.rot90(images, 3, [2, 3])
            logits_r270 = model(images_r270)
            probs += torch.sigmoid(logits_r270)

            # Average
            avg_probs = probs / 5.0

            # Store results
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            for img_id, prob in zip(image_ids, avg_probs_np):
                results.append({"Id": img_id, "Label": prob})

    return pd.DataFrame(results)


def run(train_loader, val_loader, test_loader):
    """
    Main execution function for training and inference.
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # 2. Model
    model = ResV2GeM(config=Config, pretrained=Config.pretrained)
    model = model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 4. Training Loop
    best_score = -float("inf")
    patience = 5  # Early stopping patience
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score, val_loss = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch}/{Config.epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Weighted AUC: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            save_path = os.path.join(Config.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # 5. Inference
    print("Starting inference...")
    best_model_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found, using current weights.")

    submission_df = predict_tta(model, test_loader, device)

    # Save Submission
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
