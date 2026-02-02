import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger, AverageMeter, ProbabilisticF1
from library.model import SiameseEfficientNetFPN
from library.data import get_dataloaders

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over batches
    for batch_idx, (target_img, contra_img, labels) in enumerate(loader):
        # Move to device
        target_img = target_img.to(device, non_blocking=True)
        contra_img = contra_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (with AMP if enabled)
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

        # Backward pass
        # Note: Gradient clipping is explicitly disabled per strategy
        if Config.USE_AMP:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), target_img.size(0))

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}] "
                f"Loss: {losses.val:.6f} ({losses.avg:.6f})"
            )

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Probabilistic F1.
    """
    model.eval()
    losses = AverageMeter()
    pf1_metric = ProbabilisticF1()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for target_img, contra_img, labels in loader:
            target_img = target_img.to(device, non_blocking=True)
            contra_img = contra_img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                logits = model(target_img, contra_img)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            losses.update(loss.item(), target_img.size(0))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # Compute pF1
    score = pf1_metric(y_pred, y_true)

    return losses.avg, score


def predict_and_submit(model, loader, device):
    """
    Generates predictions for the test set and saves submission file.
    Aggregates multiple views per prediction_id using MAX.
    """
    logger.info("Starting inference on test set...")
    model.eval()

    prediction_ids = []
    probabilities = []

    with torch.no_grad():
        for target_img, contra_img, pred_ids in loader:
            target_img = target_img.to(device, non_blocking=True)
            contra_img = contra_img.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                logits = model(target_img, contra_img)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            prediction_ids.extend(pred_ids)
            probabilities.extend(probs)

    # Create DataFrame
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": probabilities})

    # Aggregation: Max probability across views for the same prediction_id
    # (e.g. 10116_L might have CC and MLO views)
    submission_df = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # Save
    submission_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Head:\n{submission_df.head()}")


def run(debug=False):
    """
    Main execution function.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cache=True
    )

    # 2. Model
    logger.info(f"Initializing model: {Config.BACKBONE}")
    model = SiameseEfficientNetFPN(
        backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    # 3. Optimization
    # Weighted Loss for Imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_pf1 = -1.0
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch} Summary: "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1} | "
            f"LR: {current_lr}"
        )

        # Early Stopping & Checkpointing
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model found! pF1: {best_pf1} -> Saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # 5. Inference
    logger.info("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning("Best model file not found, using current model state.")

    predict_and_submit(model, test_loader, device)
