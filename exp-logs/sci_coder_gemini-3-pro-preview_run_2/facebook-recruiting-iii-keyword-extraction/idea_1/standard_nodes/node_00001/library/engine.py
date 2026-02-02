import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import csv
from library import config
from library import utils


def train_one_epoch(model, dataloader, optimizer, device, criterion, scaler):
    """
    Trains the model for one epoch using mixed precision.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to train on.
        criterion (nn.Module): Loss function.
        scaler (GradScaler): Gradient scaler for AMP.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch: input_ids, lengths, labels, ids
        input_ids, lengths, labels, _ = batch

        input_ids = input_ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to evaluate on.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: (average_loss, all_probabilities, all_targets)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, lengths, labels, _ = batch

            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_probs = np.array([])
        all_targets = np.array([])

    return avg_loss, all_probs, all_targets


def train_model(model, train_loader, val_loader, device):
    """
    Orchestrates the training loop, evaluation, and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train on.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion, scaler
        )
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, device, criterion
        )

        # Calculate F1 score for monitoring
        # Using the default threshold from config
        val_preds_bin = (val_probs > config.PREDICTION_THRESHOLD).astype(int)
        val_f1 = utils.calculate_f1_score(val_targets, val_preds_bin)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1: {val_f1}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")


def generate_submission(model, test_loader, tag_encoder, device):
    """
    Generates predictions for the test set and saves them to submission.csv.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        tag_encoder (TagEncoder): Encoder to decode predicted indices to strings.
        device (torch.device): Device to run inference on.
    """
    print("Generating submission...")

    # Load best model weights if available
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading weights from {config.MODEL_SAVE_PATH}")
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: No saved model found. Using current model weights.")

    model.eval()
    ids_list = []
    tags_list = []

    with torch.no_grad():
        for batch in test_loader:
            # Test loader collate returns: input_ids, lengths, ids
            input_ids, lengths, ids = batch

            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                logits = model(input_ids, lengths)

            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            ids_np = ids.cpu().numpy()

            # Decode predictions for each sample in batch
            for i in range(len(ids_np)):
                q_id = ids_np[i]
                # Decode using threshold defined in config
                pred_tags = tag_encoder.decode(
                    probs_np[i], threshold=config.PREDICTION_THRESHOLD
                )
                tags_str = " ".join(pred_tags)

                ids_list.append(q_id)
                tags_list.append(tags_str)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": ids_list, "Tags": tags_list})

    # Save to CSV
    # quoting=csv.QUOTE_NONNUMERIC ensures non-numeric fields (Tags) are quoted as per sample format
    submission_df.to_csv(
        config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Submission saved to {config.SUBMISSION_PATH}")
