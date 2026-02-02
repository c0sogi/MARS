import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import setup_logger, compute_metric

# Initialize logger
logger = setup_logger("engine")


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Handles the training of the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training set.
        optimizer: Optimizer instance.
        device: Device to run training on (CPU/GPU).
        criterion: Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = []

    for batch in dataloader:
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        scalar_features = batch["scalar_features"].to(device)
        labels = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            scalar_features,
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def validate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation set.
        device: Device to run evaluation on.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, metric_score)
    """
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )

            loss = criterion(logits, labels)
            losses.append(loss.item())

            # Apply softmax for probabilities
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = np.mean(losses)

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    # Compute metric (Log Loss)
    metric_score = compute_metric(y_true, y_pred)

    return avg_loss, metric_score


def run_training(
    model, train_loader, val_loader, epochs=Config.epochs, device=Config.device
):
    """
    Orchestrates the training process: optimizer setup, epoch loop, validation,
    early stopping, and model saving.
    """
    model.to(device)

    # Differential Learning Rates
    # Assign lower LR to backbone and higher LR to the new head
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.lr_head,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device, criterion)

        # Log results (Full precision for metric as requested)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Metric: {val_metric}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(
                f"Validation loss improved. Saving model to {Config.model_save_path}"
            )
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")


def run_inference(model, test_loader, device=Config.device):
    """
    Runs inference on the test set and saves the submission file.
    """
    # Load best weights if available
    if os.path.exists(Config.model_save_path):
        logger.info(f"Loading best model weights from {Config.model_save_path}")
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    else:
        logger.warning("No saved model found. Using current weights for inference.")

    model.to(device)
    model.eval()

    all_probs = []

    logger.info("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions
    final_probs = np.vstack(all_probs)

    # Load test metadata to retrieve IDs
    df_test = pd.read_csv(Config.test_path)

    # Handle Debug Mode alignment
    if hasattr(Config, "debug") and Config.debug:
        logger.info("Debug mode: Subsetting test IDs to match inference.")
        df_test = df_test.head(50)

    # Safety check
    if len(df_test) != len(final_probs):
        logger.error(
            f"Mismatch: Test IDs ({len(df_test)}) vs Predictions ({len(final_probs)})"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "id": df_test["id"],
            "winner_model_a": final_probs[:, 0],
            "winner_model_b": final_probs[:, 1],
            "winner_tie": final_probs[:, 2],
        }
    )

    # Save
    logger.info(f"Saving submission file to {Config.submission_path}")
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission.to_csv(Config.submission_path, index=False)
    logger.info("Submission generation complete.")
