import os
import torch
import torch.nn as nn
import numpy as np
from library import config, utils, model


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, (cat_feats, num_feats), labels in loader:
        images = images.to(device)
        cat_feats = cat_feats.to(device)
        num_feats = num_feats.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.amp.autocast("cuda"):
            logits = model(images, (cat_feats, num_feats))

            # FP32-Guarded Loss Calculation
            # This prevents NaNs when using high pos_weight in BCEWithLogitsLoss
            if config.USE_FP32_LOSS:
                with torch.amp.autocast("cuda", enabled=False):
                    loss = criterion(logits.float(), labels.float())
            else:
                loss = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation and calculates pF1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, (cat_feats, num_feats), labels in loader:
            images = images.to(device)
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda"):
                logits = model(images, (cat_feats, num_feats))

                # FP32-Guarded Loss for consistency
                if config.USE_FP32_LOSS:
                    with torch.amp.autocast("cuda", enabled=False):
                        loss = criterion(logits.float(), labels.float())
                else:
                    loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)

            running_loss += loss.item()
            all_preds.extend(probs.float().cpu().numpy().flatten())
            all_labels.extend(labels.float().cpu().numpy().flatten())

    avg_loss = running_loss / len(loader)
    pf1 = utils.pf1_score(all_labels, all_preds)

    return avg_loss, pf1


def run_training(
    train_loader, val_loader, feature_meta, epochs=config.EPOCHS, patience=3
):
    """
    Main training loop with Early Stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        feature_meta: Dictionary containing metadata like vocab_sizes.
        epochs: Maximum number of epochs.
        patience: Number of epochs to wait for improvement before stopping.

    Returns:
        str: Path to the best saved model.
    """
    device = config.DEVICE
    vocab_sizes = feature_meta["vocab_sizes"]

    # Initialize Model
    net = model.MRHNModel(vocab_sizes).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # Loss Function
    # High positive weight to counter class imbalance
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")

    # Tracking
    best_pf1 = -1.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            net, train_loader, optimizer, scheduler, scaler, criterion, device
        )
        val_loss, val_pf1 = validate_epoch(net, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(net.state_dict(), best_model_path)
            print(f"Saved Best Model (pF1: {best_pf1})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    return best_model_path
