import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger, probabilistic_f1, save_checkpoint

# Initialize logger
logger = get_logger(name="engine")


def train_one_epoch(model, loader, optimizer, scaler, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Loss functions
    # Primary: Cancer (Binary)
    criterion_cancer = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device)
    )
    # Auxiliary: Multiclass
    criterion_aux = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)
        tabular = {
            k: v.to(device, non_blocking=True) for k, v in batch["tabular"].items()
        }

        targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)
        aux_birads = batch["aux_birads"].to(device, non_blocking=True)
        aux_density = batch["aux_density"].to(device, non_blocking=True)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda"):
            outputs = model(images, tabular)

            # Calculate individual losses
            loss_cancer = criterion_cancer(outputs["cancer"], targets)
            loss_birads = criterion_aux(outputs["birads"], aux_birads)
            loss_density = criterion_aux(outputs["density"], aux_density)

            # Weighted Sum
            total_loss = (
                loss_cancer * Config.LOSS_WEIGHTS["cancer"]
                + loss_birads * Config.LOSS_WEIGHTS["birads"]
                + loss_density * Config.LOSS_WEIGHTS["density"]
            )

        # Backward Pass with Scaler
        scaler.scale(total_loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Step
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and pF1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Loss functions
    criterion_cancer = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device)
    )
    criterion_aux = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tabular = {
                k: v.to(device, non_blocking=True) for k, v in batch["tabular"].items()
            }

            targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)
            aux_birads = batch["aux_birads"].to(device, non_blocking=True)
            aux_density = batch["aux_density"].to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward (No AMP needed for eval usually, but consistent behavior is good)
            # Using float32 for stability in validation metrics
            outputs = model(images, tabular)

            loss_cancer = criterion_cancer(outputs["cancer"], targets)
            loss_birads = criterion_aux(outputs["birads"], aux_birads)
            loss_density = criterion_aux(outputs["density"], aux_density)

            total_loss = (
                loss_cancer * Config.LOSS_WEIGHTS["cancer"]
                + loss_birads * Config.LOSS_WEIGHTS["birads"]
                + loss_density * Config.LOSS_WEIGHTS["density"]
            )

            running_loss += total_loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions for metric calculation
            # Apply sigmoid to cancer logits to get probabilities
            probs = torch.sigmoid(outputs["cancer"])

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate pF1
    pf1_score = probabilistic_f1(all_targets, all_preds)

    return avg_loss, pf1_score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    device="cuda",
    epochs=Config.EPOCHS,
    patience=3,
):
    """
    Main training loop with Early Stopping.
    """
    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    scaler = torch.amp.GradScaler("cuda")
    best_score = -np.inf
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            # Check if scheduler expects metrics (e.g., ReduceLROnPlateau)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = Config.LEARNING_RATE

        # Logging (Full precision)
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1}"
        )

        # Early Stopping & Checkpointing
        if val_pf1 > best_score:
            best_score = val_pf1
            patience_counter = 0
            logger.info(
                f"New best score: {best_score}. Saving checkpoint to {best_model_path}"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, best_model_path
            )
        else:
            patience_counter += 1
            logger.info(
                f"Score did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation pF1: {best_score}")
    return best_score


def inference(model, test_loader, device="cuda"):
    """
    Generates predictions for the test set and saves the submission file.
    Aggregates predictions by prediction_id using Max pooling.
    """
    logger.info("Starting inference on test set...")
    model.eval()

    prediction_ids = []
    cancer_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            tabular = {
                k: v.to(device, non_blocking=True) for k, v in batch["tabular"].items()
            }
            batch_ids = batch["prediction_id"]

            # Forward
            outputs = model(images, tabular)

            # Get probabilities
            probs = torch.sigmoid(outputs["cancer"]).cpu().numpy().flatten()

            prediction_ids.extend(batch_ids)
            cancer_probs.extend(probs)

    # Create DataFrame
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": cancer_probs})

    logger.info(f"Raw predictions generated: {len(df_pred)} rows.")

    # Aggregate by prediction_id (Max Pooling)
    # Idea 6 specifies taking the maximum probability across all images for a prediction_id
    submission_df = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # Save submission
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info("Submission saved successfully.")
    return submission_df
