import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import get_logger, competition_metric
from library.loss import HybridLoss

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False, disable=True)

    for batch in pbar:
        # Unpack batch
        images = batch["images"].to(device, dtype=torch.float32)
        study_targets = batch["study_targets"].to(device, dtype=torch.float32)
        slice_targets = batch["slice_targets"].to(device, dtype=torch.float32)
        slice_mask = batch["slice_mask"].to(device, dtype=torch.float32)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            study_logits, slice_logits = model(images)
            loss = criterion(
                study_logits, study_targets, slice_logits, slice_targets, slice_mask
            )

        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Update progress bar
        pbar.set_postfix(loss=running_loss / dataset_size)

    epoch_loss = running_loss / dataset_size

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Store predictions and targets for metric calculation
    all_study_preds = []
    all_study_targets = []

    pbar = tqdm(loader, desc="[Val]", leave=False, disable=True)

    with torch.no_grad():
        for batch in pbar:
            images = batch["images"].to(device, dtype=torch.float32)
            study_targets = batch["study_targets"].to(device, dtype=torch.float32)
            slice_targets = batch["slice_targets"].to(device, dtype=torch.float32)
            slice_mask = batch["slice_mask"].to(device, dtype=torch.float32)

            batch_size = images.size(0)

            # Forward pass (no autocast needed for eval usually, but good for consistency)
            study_logits, slice_logits = model(images)
            loss = criterion(
                study_logits, study_targets, slice_logits, slice_targets, slice_mask
            )

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to study logits for metric calculation
            study_probs = torch.sigmoid(study_logits)

            all_study_preds.append(study_probs.cpu().numpy())
            all_study_targets.append(study_targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_study_preds = np.concatenate(all_study_preds, axis=0)
    all_study_targets = np.concatenate(all_study_targets, axis=0)

    # Calculate Competition Metric
    metric_score = competition_metric(all_study_preds, all_study_targets)

    return epoch_loss, metric_score


def fit(model, train_loader, val_loader, epochs=Config.EPOCHS, device=Config.DEVICE):
    """
    Main training loop with Early Stopping.
    """
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = HybridLoss().to(device)
    scaler = GradScaler()

    best_metric = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val Metric: {val_metric:.10f}"  # Full precision as requested
        )

        # Early Stopping & Model Checkpointing
        # We monitor the competition metric (lower is better)
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(f"New best model saved with metric: {best_metric:.10f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Metric: {best_metric:.10f}")


def predict(model, test_loader, device=Config.DEVICE):
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Starting prediction on test set...")

    # Load best model weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        logger.info(f"Loaded weights from {Config.MODEL_PATH}")
    else:
        logger.warning("No saved model found. Using current model weights.")

    model.to(device)
    model.eval()

    # Dictionary to store predictions: {StudyInstanceUID: [p_C1, ..., p_Overall]}
    predictions_map = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Predict]", disable=True):
            images = batch["images"].to(device, dtype=torch.float32)
            uids = batch["row_id"]  # List of strings

            # Forward pass
            study_logits, _ = model(images)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()

            # Store in map
            for i, uid in enumerate(uids):
                predictions_map[uid] = study_probs[i]

    # Generate Submission DataFrame
    # We need to map the 8 outputs to the specific rows requested in test.csv
    # Config.TARGET_COLUMNS order: ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Load test.csv to get the required row_ids
    test_csv_path = os.path.join(Config.INPUT_ROOT, "test.csv")
    if not os.path.exists(test_csv_path):
        # Fallback to sample submission if test.csv is not available (e.g. in some environments)
        logger.warning("test.csv not found, using sample_submission.csv structure.")
        test_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        # We need to parse row_id to get UID and prediction_type
        # row_id format: UID_prediction_type
        # However, sample submission just has row_id. We need to split it.
        # This is slightly risky if UID contains underscores, but DICOM UIDs usually use dots.
        # A safer way is to match against known suffixes.
        pass
    else:
        test_df = pd.read_csv(test_csv_path)

    # Column mapping
    col_to_idx = {col: i for i, col in enumerate(Config.TARGET_COLUMNS)}

    results = []

    # Iterate over required rows
    # If using test.csv, we have 'StudyInstanceUID' and 'prediction_type'
    if "prediction_type" in test_df.columns:
        for _, row in test_df.iterrows():
            row_id = row["row_id"]
            uid = row["StudyInstanceUID"]
            pred_type = row["prediction_type"]

            if uid in predictions_map:
                idx = col_to_idx.get(pred_type)
                if idx is not None:
                    prob = predictions_map[uid][idx]
                else:
                    prob = 0.5  # Fallback
            else:
                prob = 0.5  # Fallback for missing studies

            results.append({"row_id": row_id, "fractured": prob})

    else:
        # Fallback parsing for sample_submission format
        for _, row in test_df.iterrows():
            row_id = row["row_id"]
            # Find which target column matches the suffix
            matched = False
            for target in Config.TARGET_COLUMNS:
                if row_id.endswith(f"_{target}"):
                    # Extract UID
                    uid = row_id[: -(len(target) + 1)]

                    if uid in predictions_map:
                        idx = col_to_idx[target]
                        prob = predictions_map[uid][idx]
                    else:
                        prob = 0.5

                    results.append({"row_id": row_id, "fractured": prob})
                    matched = True
                    break
            if not matched:
                results.append({"row_id": row_id, "fractured": 0.5})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
