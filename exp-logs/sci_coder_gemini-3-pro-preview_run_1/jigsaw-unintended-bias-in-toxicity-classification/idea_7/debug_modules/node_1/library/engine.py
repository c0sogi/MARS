import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import get_logger
from library.loss import HybridLoss
from library.awp import AWP
from library.metrics import calculate_final_score

logger = get_logger()


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, awp=None):
    """
    Trains the model for one epoch using Mixed Precision and optional AWP.
    """
    model.train()
    scaler = GradScaler(enabled=Config.FP16)
    criterion = HybridLoss()

    dataset_size = 0
    running_loss = 0.0

    # Set gradients to zero before starting the loop
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)

        targets = batch["target"].to(device, non_blocking=True)
        aux_targets = batch["aux_targets"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        batch_size = ids.size(0)

        # Prepare batch dict for loss function
        batch_data = {"target": targets, "aux_targets": aux_targets, "weight": weights}

        # 1. Standard Forward Pass
        with autocast(enabled=Config.FP16):
            outputs = model(ids, mask, token_type_ids)
            loss_dict = criterion(outputs, batch_data)
            loss = loss_dict["loss"]
            # Scale loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # 2. Standard Backward Pass
        scaler.scale(loss).backward()

        # Accumulate metrics
        running_loss += (loss.item() * Config.ACCUMULATION_STEPS) * batch_size
        dataset_size += batch_size

        # 3. Optimization Step (performed every ACCUMULATION_STEPS)
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:

            # 4. Adversarial Weight Perturbation (AWP)
            if Config.USE_AWP and awp is not None and epoch >= Config.AWP_START_EPOCH:
                # Save weights and apply perturbation based on current gradients
                awp.attack_step()

                # Forward pass with perturbed weights
                with autocast(enabled=Config.FP16):
                    adv_outputs = model(ids, mask, token_type_ids)
                    adv_loss_dict = criterion(adv_outputs, batch_data)
                    adv_loss = adv_loss_dict["loss"]
                    # Scale adversarial loss
                    adv_loss = adv_loss / Config.ACCUMULATION_STEPS

                # Backward pass with perturbed weights (accumulate gradients)
                scaler.scale(adv_loss).backward()

                # Restore original weights
                awp._restore()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    epoch_loss = running_loss / dataset_size
    logger.info(f"Train Epoch {epoch} | Loss: {epoch_loss}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss


def validate(model, dataloader, device, val_df):
    """
    Evaluates the model on the validation set and computes the competition metric.
    """
    model.eval()
    criterion = HybridLoss()

    dataset_size = 0
    running_loss = 0.0

    # Store predictions for metric calculation
    # Using a list of numpy arrays is efficient for concatenation later
    all_preds = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)

            targets = batch["target"].to(device, non_blocking=True)
            aux_targets = batch["aux_targets"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            batch_size = ids.size(0)

            batch_data = {
                "target": targets,
                "aux_targets": aux_targets,
                "weight": weights,
            }

            # Forward Pass (No AWP, No Mixed Precision needed for Val usually, but safe to use)
            with autocast(enabled=Config.FP16):
                outputs = model(ids, mask, token_type_ids)
                loss_dict = criterion(outputs, batch_data)
                loss = loss_dict["loss"]

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Extract toxicity logits and apply sigmoid for probability
            tox_logits = outputs["toxicity_logits"]
            probs = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
            all_preds.append(probs)

    epoch_loss = running_loss / dataset_size

    # Concatenate all predictions
    predictions = np.concatenate(all_preds)

    # Calculate Competition Metric
    # Note: val_df must align with the dataloader order.
    # The dataloader in data.py uses SequentialSampler for validation, so order is preserved.
    final_score, metrics_dict = calculate_final_score(val_df, predictions)

    logger.info(f"Validation Loss: {epoch_loss}")
    logger.info(f"Validation Score: {final_score}")

    # Print detailed metrics with full precision
    logger.info("Detailed Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")

    return final_score, epoch_loss


def inference(model, dataloader, device, submission_path="./submission/submission.csv"):
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Starting Inference...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)

            with autocast(enabled=Config.FP16):
                outputs = model(ids, mask, token_type_ids)

            tox_logits = outputs["toxicity_logits"]
            probs = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
            all_preds.append(probs)

    predictions = np.concatenate(all_preds)

    # Load sample submission to get IDs and correct format
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(predictions) != len(sample_sub):
        logger.warning(
            f"Prediction length {len(predictions)} does not match sample submission {len(sample_sub)}"
        )

    sample_sub["prediction"] = predictions

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    sample_sub.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
