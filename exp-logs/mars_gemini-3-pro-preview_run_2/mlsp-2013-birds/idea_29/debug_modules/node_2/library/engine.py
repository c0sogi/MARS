import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import compute_auc
from library.dataset import BirdDataset


def mixup_data(x, y_hard, y_soft, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to inputs and prepares pairs of targets.
    Returns mixed inputs, pairs of hard/soft targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_hard_a, y_hard_b = y_hard, y_hard[index]
    y_soft_a, y_soft_b = y_soft, y_soft[index]
    return mixed_x, y_hard_a, y_hard_b, y_soft_a, y_soft_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, device, pos_weights):
    """
    Executes one epoch of training.
    Handles Mixup and the Weighted Distillation Loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # BCEWithLogitsLoss with class balancing weights
    # pos_weights should be on the correct device
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    for batch in loader:
        images = batch["image"].to(device, dtype=torch.float32)
        labels = batch["labels"].to(device, dtype=torch.float32)
        soft_labels = batch["soft_labels"].to(device, dtype=torch.float32)

        batch_size = images.size(0)
        dataset_size += batch_size

        # Apply Mixup if enabled in Config
        if Config.MIXUP_ALPHA > 0:
            images, labels_a, labels_b, soft_a, soft_b, lam = mixup_data(
                images, labels, soft_labels, Config.MIXUP_ALPHA, device
            )

            logits = model(images)

            # Compute losses for both mixed components
            loss_hard = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss_soft = mixup_criterion(criterion, logits, soft_a, soft_b, lam)
        else:
            logits = model(images)
            loss_hard = criterion(logits, labels)
            loss_soft = criterion(logits, soft_labels)

        # Weighted Distillation Loss
        # L = BCE(Target, Pred) + lambda * BCE(TTA_Targets, Pred)
        # Note: pos_weight is applied within the criterion
        loss = loss_hard + Config.DISTILLATION_LAMBDA * loss_soft

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, device, pos_weights):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds_list = []
    targets_list = []

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            labels = batch["labels"].to(device, dtype=torch.float32)

            batch_size = images.size(0)
            dataset_size += batch_size

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size

            preds_list.append(torch.sigmoid(logits).cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    auc_score = compute_auc(all_targets, all_preds)

    return avg_loss, auc_score


def inference_with_tta(model, df, device, batch_size=32, num_workers=4):
    """
    Performs Cyclic Test-Time Augmentation (TTA) inference.
    Iterates through defined time-shifts (Original + 3 shifts), generates predictions,
    and returns the averaged probabilities.
    """
    model.eval()

    num_samples = len(df)
    num_species = Config.NUM_SPECIES
    accumulated_probs = np.zeros((num_samples, num_species), dtype=np.float32)

    # Define shifts based on Config (e.g., 0.0, 0.25, 0.50, 0.75)
    shifts = np.linspace(0, 1, Config.TTA_STEPS, endpoint=False)

    model.to(device)

    with torch.no_grad():
        for shift in shifts:
            # Create a dataset instance with the specific fixed shift
            # We reuse the cache since the shift is applied in __getitem__
            ds = BirdDataset(
                df=df,
                phase="test",
                model_name=model.model_name,
                load_cached_data=True,
                fixed_shift_ratio=shift,
            )

            # Use a DataLoader with shuffle=False to ensure order is preserved
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            probs_list = []
            for batch in loader:
                images = batch["image"].to(device, dtype=torch.float32)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)

            # Concatenate predictions for the current shift
            shift_probs = np.concatenate(probs_list, axis=0)
            accumulated_probs += shift_probs

    # Average predictions across all TTA steps
    avg_probs = accumulated_probs / Config.TTA_STEPS

    return avg_probs


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    pos_weights,
    num_epochs,
    patience,
    save_path,
):
    """
    Orchestrates the full training loop with Early Stopping.
    Saves the best model based on Validation AUC.
    """
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs with patience {patience}...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, pos_weights
        )
        val_loss, val_auc = evaluate(model, val_loader, device, pos_weights)

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping Logic (Maximize AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load and return the best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
