import torch
import numpy as np
from library.utils import calculate_roc_auc, save_checkpoint


def train_fn(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    criterion,
    epochs,
    patience,
    save_path,
):
    """
    Runs the full training loop with gradient scaling, backpropagation, and early stopping.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: 'cuda' or 'cpu'.
        criterion: Loss function.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model checkpoint.

    Returns:
        float: Best validation AUC score achieved.
    """
    # Initialize Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_losses = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

            optimizer.zero_grad()

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with Scaler
            scaler.scale(loss).backward()
            scaler.scale(optimizer).step()
            scaler.update()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        val_loss, val_auc, _ = eval_fn(model, val_loader, device, criterion)

        # --- Scheduler Step ---
        if scheduler is not None:
            scheduler.step()

        # --- Logging ---
        # Print full precision metrics as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save the best model state
            save_checkpoint(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_auc


def eval_fn(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: 'cuda' or 'cpu'.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, auc_score, predictions)
    """
    model.eval()
    losses = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            # Apply Sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate ROC AUC
    auc = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc, all_preds


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: Test DataLoader (returns inputs and clip names).
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (clip_names, probabilities)
    """
    model.eval()
    all_preds = []
    all_clips = []

    with torch.no_grad():
        for batch in dataloader:
            # Test dataset returns (inputs, clip_name)
            inputs, clips = batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_clips.extend(clips)

    all_preds = np.concatenate(all_preds)

    # Flatten predictions if they are (N, 1) to (N,)
    if all_preds.ndim > 1 and all_preds.shape[1] == 1:
        all_preds = all_preds.flatten()

    return all_clips, all_preds
