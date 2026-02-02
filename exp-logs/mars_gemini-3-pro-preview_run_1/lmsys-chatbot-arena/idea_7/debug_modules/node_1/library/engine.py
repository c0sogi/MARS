import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using gradient accumulation.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to run on.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # CrossEntropyLoss works with soft targets (probabilities) since PyTorch 1.10
    criterion = nn.CrossEntropyLoss()

    for step, data in enumerate(dataloader):
        # Move inputs to device
        ids_a = data["input_ids_a"].to(device)
        mask_a = data["attention_mask_a"].to(device)
        ids_b = data["input_ids_b"].to(device)
        mask_b = data["attention_mask_b"].to(device)
        struct_feats = data["structural_features"].to(device)
        targets = data["labels"].to(device)

        batch_size = ids_a.size(0)

        # Forward pass
        outputs = model(ids_a, mask_a, ids_b, mask_b, struct_feats)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Normalize loss for gradient accumulation
        loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        loss.backward()

        # Update weights if accumulation steps reached
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Accumulate running loss (scale back up for logging)
        running_loss += (loss.item() * Config.GRAD_ACCUM_STEPS) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Train Epoch: {epoch} | Loss: {epoch_loss}")
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: The device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in dataloader:
            ids_a = data["input_ids_a"].to(device)
            mask_a = data["attention_mask_a"].to(device)
            ids_b = data["input_ids_b"].to(device)
            mask_b = data["attention_mask_b"].to(device)
            struct_feats = data["structural_features"].to(device)
            targets = data["labels"].to(device)

            batch_size = ids_a.size(0)

            outputs = model(ids_a, mask_a, ids_b, mask_b, struct_feats)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    # Print full precision as requested
    print(f"Validation Loss: {epoch_loss}")
    return epoch_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Orchestrates the full training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to run on.
        num_epochs: Maximum number of epochs.
        patience: Patience for early stopping.

    Returns:
        model: The model with the best weights loaded.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_OUTPUT_PATH

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # Load best model for future use
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        device: The device to run on.
    """
    model.eval()
    all_probs = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for data in test_loader:
            ids_a = data["input_ids_a"].to(device)
            mask_a = data["attention_mask_a"].to(device)
            ids_b = data["input_ids_b"].to(device)
            mask_b = data["attention_mask_b"].to(device)
            struct_feats = data["structural_features"].to(device)

            outputs = model(ids_a, mask_a, ids_b, mask_b, struct_feats)

            # Convert logits to probabilities using Softmax
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)

    # Load test metadata to get IDs
    try:
        test_df = pd.read_csv(Config.TEST_PATH)
        if Config.DEBUG:
            test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)
        submission_ids = test_df["id"]
    except Exception as e:
        print(f"Error reading test metadata for IDs: {e}")
        return

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": submission_ids,
            "winner_model_a": all_probs[:, 0],
            "winner_model_b": all_probs[:, 1],
            "winner_tie": all_probs[:, 2],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    print(submission_df.head())
