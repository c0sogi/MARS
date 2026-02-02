import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import DEVICE, MODEL_SAVE_PATH, SUBMISSION_FILE
from library.utils import save_submission


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move data to device
        cat_x = batch["cat"].to(device)
        cont_x = batch["cont"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(cat_x, cont_x)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Step scheduler (OneCycleLR updates per batch)
        if scheduler is not None:
            scheduler.step()

        # Accumulate loss
        running_loss += loss.item() * targets.size(0)
        dataset_size += targets.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cat_x = batch["cat"].to(device)
            cont_x = batch["cont"].to(device)
            targets = batch["target"].to(device)

            logits = model(cat_x, cont_x)
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * targets.size(0)
            dataset_size += targets.size(0)

            # Collect targets and predictions for metric calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate and flatten arrays
    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    # Calculate ROC AUC
    auc_score = roc_auc_score(all_targets, all_preds)

    return epoch_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The trained PyTorch model.
        dataloader: Test DataLoader.
        device: Device to run inference on.

    Returns:
        tuple: (Array of IDs, Array of predicted probabilities)
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["id"]
            cat_x = batch["cat"].to(device)
            cont_x = batch["cont"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits)

            ids_list.append(ids)
            preds_list.append(probs.cpu().numpy())

    # Concatenate results
    all_ids = np.concatenate(ids_list)
    all_preds = np.concatenate(preds_list).flatten()

    return all_ids, all_preds


def train_engine(
    model, train_loader, val_loader, test_loader, epochs, max_lr, weight_decay, patience
):
    """
    Orchestrates the training process, evaluation, early stopping, and final prediction.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        epochs: Maximum number of epochs.
        max_lr: Maximum learning rate for OneCycleLR.
        weight_decay: Weight decay for AdamW.
        patience: Patience for early stopping.
    """
    # Move model to device
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    best_auc = -float("inf")
    patience_counter = 0

    print(f"Starting training on {DEVICE}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE
        )

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model for inference
    print("Loading best model for final predictions...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # Generate predictions on test set
    print("Generating submission...")
    ids, preds = predict(model, test_loader, DEVICE)

    # Save submission file
    save_submission(ids, preds, SUBMISSION_FILE)
    print("Process complete.")
