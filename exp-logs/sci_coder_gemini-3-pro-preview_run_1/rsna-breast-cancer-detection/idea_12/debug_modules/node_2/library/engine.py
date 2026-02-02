import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import probabilistic_f1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move data to device
        target = batch["target"].to(device)
        contra = batch["contra"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # Shape (B, 1)

        batch_size = target.size(0)
        dataset_size += batch_size

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(target, contra)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step (No gradient clipping enforced)
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (average_loss, pF1_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            target = batch["target"].to(device)
            contra = batch["contra"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            batch_size = target.size(0)
            dataset_size += batch_size

            # Forward pass
            logits = model(target, contra)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        # Compute Probabilistic F1
        pf1 = probabilistic_f1(all_labels, all_probs)
    else:
        pf1 = 0.0

    return epoch_loss, pf1


def fit(model, train_loader, val_loader, device, epochs=Config.NUM_EPOCHS, patience=3):
    """
    Main training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        model: The best model (loaded from checkpoint).
    """
    # Define Loss with positive weight for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_pf1 = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics (Full precision)
        print(
            f"Epoch {epoch + 1}/{epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        # Early Stopping Logic
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! pF1: {best_pf1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model state
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def inference(model, test_loader, device):
    """
    Generates predictions for the test set, aggregates them by prediction_id,
    and saves the submission file.

    Args:
        model: The trained PyTorch model.
        test_loader: Test DataLoader.
        device: Torch device.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()
    results = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            target = batch["target"].to(device)
            contra = batch["contra"].to(device)
            prediction_ids = batch["prediction_id"]

            # Forward pass
            logits = model(target, contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Collect results
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create raw dataframe
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("Warning: No predictions generated.")
        # Create empty submission with correct columns just in case
        submission_df = pd.DataFrame(columns=["prediction_id", "cancer"])
    else:
        # Aggregate: Max probability per prediction_id (Breast Level)
        # As per task description: "Multiple images will share the same prediction ID."
        submission_df = df_results.groupby("prediction_id", as_index=False)[
            "cancer"
        ].max()

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
