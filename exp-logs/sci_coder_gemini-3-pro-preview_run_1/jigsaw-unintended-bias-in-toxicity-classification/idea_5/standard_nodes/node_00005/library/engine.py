import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import get_logger, AverageMeter, get_device
from library.loss import TriangulationLoss
from library.metrics import calculate_score
from library.model import TriangulationDeberta

logger = get_logger(name="engine")


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, criterion):
    """
    Handles the training of one epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    primary_loss_meter = AverageMeter()

    # Iterate over dataloader
    # No tqdm as per instructions
    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Prepare targets and weights for loss calculation
        # The loss function expects these in the batch dict on the correct device
        batch_gpu = {
            "target": batch["target"].to(device),
            "identity_targets": batch["identity_targets"].to(device),
            "attack_target": batch["attack_target"].to(device),
            "sample_weight": batch["sample_weight"].to(device),
        }

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss
        loss, loss_dict = criterion(outputs, batch_gpu)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        # Update metrics
        batch_size = input_ids.size(0)
        loss_meter.update(loss.item(), batch_size)
        primary_loss_meter.update(loss_dict["loss_primary"], batch_size)

        # Optional: Log every N steps if needed, but keeping silent as requested

    logger.info(
        f"Epoch {epoch+1} Train Loss: {loss_meter.avg:.6f} (Primary: {primary_loss_meter.avg:.6f})"
    )

    return loss_meter.avg


def validate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the competition metric score.
    """
    model.eval()

    loss_meter = AverageMeter()
    preds = []

    # We need the validation metadata to calculate the score
    # We assume the dataloader is sequential and matches the metadata file order
    df_val = pd.read_csv(Config.VAL_META_PATH)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prepare targets for loss calculation
            batch_gpu = {
                "target": batch["target"].to(device),
                "identity_targets": batch["identity_targets"].to(device),
                "attack_target": batch["attack_target"].to(device),
                "sample_weight": batch["sample_weight"].to(device),
            }

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Calculate Loss
            loss, _ = criterion(outputs, batch_gpu)
            loss_meter.update(loss.item(), input_ids.size(0))

            # Collect predictions from the Primary Head
            # Apply sigmoid to get probabilities
            primary_logits = outputs["primary"]
            probs = torch.sigmoid(primary_logits).cpu().numpy().flatten()
            preds.extend(probs)

    # Calculate Competition Score
    # Assign predictions to the dataframe
    # Ensure lengths match
    if len(preds) != len(df_val):
        logger.warning(
            f"Validation prediction count ({len(preds)}) matches dataframe ({len(df_val)})?"
        )
        # Truncate or pad if necessary (should not happen with correct loaders)
        if len(preds) > len(df_val):
            preds = preds[: len(df_val)]
        else:
            # This is a critical error state, but we handle gracefully for pipeline continuity
            logger.error("Mismatch in validation lengths.")
            return loss_meter.avg, 0.0

    df_val["prediction"] = preds

    final_score, metrics_dict = calculate_score(df_val, "prediction")

    logger.info(f"Validation Loss: {loss_meter.avg:.6f}")
    logger.info(f"Validation Score: {final_score:.6f}")
    logger.info(f"  Overall AUC: {metrics_dict['overall_auc']:.6f}")
    logger.info(f"  Subgroup AUC: {metrics_dict['subgroup_auc_mean']:.6f}")
    logger.info(f"  BPSN AUC: {metrics_dict['bpsn_auc_mean']:.6f}")
    logger.info(f"  BNSP AUC: {metrics_dict['bnsp_auc_mean']:.6f}")

    return loss_meter.avg, final_score


def run_training(train_loader, val_loader):
    """
    Main driver for training the model.
    """
    device = get_device()
    logger.info(f"Device: {device}")

    # Initialize Model
    model = TriangulationDeberta(Config.MODEL_NAME)
    model.to(device)

    # Optimizer
    # Using standard AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function
    criterion = TriangulationLoss()

    # Early Stopping Variables
    best_score = -float("inf")
    patience = 1  # Strict patience due to limited time/epochs
    patience_counter = 0
    best_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "best_model.bin")

    for epoch in range(Config.EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, criterion
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, device, criterion)

        # Checkpoint & Early Stopping
        if val_score > best_score:
            logger.info(
                f"Score improved from {best_score:.6f} to {val_score:.6f}. Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            logger.info(f"Score did not improve from {best_score:.6f}.")
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Training complete. Best Score: {best_score:.6f}")
    return best_model_path


def predict_and_submit(model_path, test_loader):
    """
    Loads the best model, generates predictions on test set, and saves submission.
    """
    device = get_device()
    logger.info("Starting prediction on Test set...")

    # Load Model
    model = TriangulationDeberta(Config.MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Get probabilities from primary head
            probs = torch.sigmoid(outputs["primary"]).cpu().numpy().flatten()
            preds.extend(probs)

    # Load Sample Submission to preserve ID order
    # The test_loader is sequential and matches the test.csv order
    submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Safety check on length
    if len(preds) != len(submission):
        logger.warning(
            f"Prediction length {len(preds)} != Submission length {len(submission)}"
        )
        # Adjust if necessary
        if len(preds) > len(submission):
            preds = preds[: len(submission)]
        else:
            # Pad with zeros (unlikely scenario)
            preds = preds + [0.0] * (len(submission) - len(preds))

    submission["prediction"] = preds

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
