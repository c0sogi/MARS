import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, scaler, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: The optimizer.
        scaler: GradScaler for AMP.
        device: 'cuda' or 'cpu'.

    Returns:
        avg_loss (float): Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Automatic Mixed Precision
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        avg_auc (float): The average ROC AUC score across all labels.
        class_aucs (dict): Dictionary of AUC scores per class.
        avg_loss (float): Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Apply sigmoid for predictions
            preds = torch.sigmoid(outputs)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) == 0:
        return 0.0, {}, avg_loss

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    class_aucs = {}
    auc_sum = 0.0
    valid_cols = 0

    # Calculate AUC for each column
    for i, col_name in enumerate(Config.TARGET_COLS):
        try:
            # Check if there is more than one class in the targets
            if len(np.unique(all_labels[:, i])) > 1:
                score = roc_auc_score(all_labels[:, i], all_preds[:, i])
            else:
                # If only one class is present (e.g., all 0s), AUC is undefined.
                # We assign 0.5 as a neutral score.
                score = 0.5
        except ValueError:
            score = 0.5

        class_aucs[col_name] = score
        auc_sum += score
        valid_cols += 1

    avg_auc = auc_sum / valid_cols if valid_cols > 0 else 0.0

    return avg_auc, class_aucs, avg_loss


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Max epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    set_seed(Config.SEED)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = -1.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_auc, val_class_aucs, val_loss = validate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC (Avg): {val_auc}")
        # Printing full precision as requested

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def predict(model, dataloader, df_test, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained PyTorch model.
        dataloader: Test DataLoader.
        df_test: DataFrame containing test metadata (must align with dataloader).
        device: Device.
        output_path: Path to save the CSV.
    """
    model.eval()
    all_preds = []

    print("Starting inference...")

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        # Fallback for empty dataloader (should not happen)
        all_preds = np.zeros((len(df_test), Config.NUM_CLASSES))

    # Create Submission DataFrame
    # Ensure columns are in the correct order as per Config.TARGET_COLS
    sub_df = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)

    # Insert StudyInstanceUID at the beginning
    sub_df.insert(0, "StudyInstanceUID", df_test["StudyInstanceUID"].values)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
