import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
from torch.amp import autocast, GradScaler
from library.config import Config
from library.utils import get_logger, calculate_weighted_loss

logger = get_logger(name="engine")


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation, BCELoss, and AMP.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Zero gradients at the start
    optimizer.zero_grad()

    # Cite debug_lesson_13: Use BCEWithLogitsLoss for stability in AMP
    criterion = nn.BCEWithLogitsLoss()

    for step, (images, targets, _) in enumerate(dataloader):
        batch_size = images.size(0)

        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        # Forward Pass with AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            # Cast to float32 for numerical stability
            outputs = outputs.to(torch.float32)
            # Outputs are logits, no clamping needed for loss
            loss = criterion(outputs, targets)
            loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Optimization Step
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics (multiply back to get actual loss value)
        running_loss += (loss.item() * Config.ACCUMULATION_STEPS) * batch_size
        dataset_size += batch_size

        # Explicit memory cleanup
        del images, targets, outputs, loss

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set and calculates the Weighted Log Loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Cite debug_lesson_13: Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets, _ in dataloader:
            batch_size = images.size(0)

            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                outputs = outputs.to(torch.float32)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to convert logits to probabilities for metrics
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            del images, targets, outputs, loss

    # Calculate Metrics
    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    weighted_metric = calculate_weighted_loss(all_targets, all_preds)

    gc.collect()

    return epoch_loss, weighted_metric


def fit(model, train_loader, val_loader, device, epochs=Config.EPOCHS):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize GradScaler for AMP
    scaler = GradScaler("cuda")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_metric = float("inf")
    patience = 3
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        val_loss, val_metric = validate(model, val_loader, device)

        scheduler.step()

        # Print full precision metrics
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val Weighted Log Loss: {val_metric}"
        )

        # Checkpoint
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0

            save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # Load best weights
    best_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        # Cite debug_lesson_8: Use weights_only=True for safety
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        logger.info("Loaded best model weights for inference.")

    return model


def inference(model, test_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()
    results = []

    # Target columns in order output by model
    # Index 0: patient_overall, Index 1-7: C1-C7
    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    logger.info("Starting inference on test set...")

    with torch.no_grad():
        for images, _, uids in test_loader:
            images = images.to(device, dtype=torch.float32)

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                outputs = outputs.to(torch.float32)
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

            probs = probs.cpu().numpy()

            # Map predictions to submission format
            for i, uid in enumerate(uids):
                row_probs = probs[i]

                for col_idx, col_name in enumerate(target_cols):
                    row_id = f"{uid}_{col_name}"
                    prob = row_probs[col_idx]
                    results.append({"row_id": row_id, "fractured": prob})

            del images, outputs

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save submission
    save_dir = "./submission"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "submission.csv")

    submission_df.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path} with {len(submission_df)} rows.")
