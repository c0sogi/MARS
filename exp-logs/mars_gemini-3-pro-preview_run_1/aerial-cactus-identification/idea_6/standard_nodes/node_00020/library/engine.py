import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import copy
import os

from library.config import Config


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Deep Supervision and Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (N, 1)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, labels, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        # Forward pass
        # RepVGGDeepSup returns (main_out, aux_out) in training mode
        main_out, aux_out = model(inputs)

        # Calculate Loss
        # Main Head Loss
        loss_main = mixup_criterion(criterion, main_out, targets_a, targets_b, lam)

        # Aux Head Loss
        loss_aux = mixup_criterion(criterion, aux_out, targets_a, targets_b, lam)

        # Total Loss
        loss = loss_main + (Config.AUX_LOSS_WEIGHT * loss_aux)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            # RepVGGDeepSup returns only main_out in eval mode (if not deployed yet)
            # or if deployed.
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0

    return epoch_loss, auc


def train_model(model, train_loader, val_loader, device):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_auc = 0.0
    best_epoch = 0
    patience = 7
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  New best model saved! AUC: {best_auc}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc} at epoch {best_epoch+1}"
            )
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def predict_tta(model, test_loader, device):
    """
    Performs inference using 4-view Test Time Augmentation (TTA).
    Views: Original, H-Flip, V-Flip, Rot180.

    Args:
        model: The trained model (should be in deploy mode).
        test_loader: DataLoader for test data (returns images, ids).
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Mapping of id -> probability
    """
    model.eval()
    predictions = {}

    # Ensure model is in deploy mode for efficiency
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip
            inputs_h = torch.flip(inputs, [3])
            out2 = torch.sigmoid(model(inputs_h))

            # 3. Vertical Flip
            inputs_v = torch.flip(inputs, [2])
            out3 = torch.sigmoid(model(inputs_v))

            # 4. Rotate 180 (H + V flip)
            inputs_r = torch.flip(inputs, [2, 3])
            out4 = torch.sigmoid(model(inputs_r))

            # Average predictions
            avg_preds = (out1 + out2 + out3 + out4) / 4.0
            avg_preds = avg_preds.cpu().numpy().flatten()

            # Store results
            for img_id, prob in zip(ids, avg_preds):
                predictions[img_id] = prob

    return predictions
