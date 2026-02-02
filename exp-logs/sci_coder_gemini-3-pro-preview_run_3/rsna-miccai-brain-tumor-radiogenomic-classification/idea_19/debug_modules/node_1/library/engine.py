import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import print_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).float()

        # Ensure targets are (B, 1) to match model output (B, 1)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect predictions for AUC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC (handle case with single class in batch)
    try:
        epoch_auc = roc_auc_score(all_targets.flatten(), all_preds.flatten())
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            if len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets.flatten(), all_preds.flatten())
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path,
    patience=5,
):
    """
    Runs the full training loop with Early Stopping.
    """
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics using the utility function
        metrics = {
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Train Loss": train_loss,
            "Train AUC": train_auc,
            "Val Loss": val_loss,
            "Val AUC": val_auc,
        }
        print_metrics(metrics)

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved! (AUC: {best_auc})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a tuple of (ids, probabilities).
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_ids.extend(ids)
            all_preds.extend(probs)

    return all_ids, all_preds
