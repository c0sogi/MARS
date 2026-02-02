import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, update_bn

from library.config import Config
from library.utils import save_checkpoint


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the input batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
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
    """
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Training logic for one epoch using Mixup and BCEWithLogitsLoss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, labels, Config.MIXUP_ALPHA, device
        )

        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Validation logic computing Loss and ROC AUC.
    Prints metrics with full precision.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute Macro Average ROC AUC
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    class_aucs = []
    for i in range(all_targets.shape[1]):
        # Calculate AUC only for classes that have both positive and negative samples
        if len(np.unique(all_targets[:, i])) == 2:
            try:
                class_aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
            except ValueError:
                continue

    auc = np.mean(class_aucs) if class_aucs else 0.5

    return epoch_loss, auc, all_preds


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    swa_start_epoch,
    save_path,
    patience=10,
):
    """
    Main training loop handling SWA activation, updates, and Early Stopping.
    """
    swa_model = AveragedModel(model)
    best_auc = 0.0
    patience_counter = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(
        f"Starting training for {epochs} epochs. SWA starts at epoch {swa_start_epoch}."
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc, _ = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        if scheduler:
            scheduler.step()

        # SWA Logic
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            # Explicitly disable Early Stopping during the SWA phase
        else:
            # Early Stopping Logic (Active only before SWA)
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                if patience is not None:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

    # Finalize SWA
    print("Updating SWA Batch Normalization statistics...")

    # Wrapper to extract images from dictionary batch for update_bn
    def swa_loader_wrapper(loader):
        for batch in loader:
            yield batch["image"]

    update_bn(swa_loader_wrapper(train_loader), swa_model, device=device)

    print(f"Saving SWA model to {save_path}")
    save_checkpoint(swa_model.state_dict(), save_path)

    return swa_model


def inference(model, test_loader, device, submission_path):
    """
    Generates predictions for the test set using Test-Time Augmentation (Horizontal Flip).
    Saves the results to the submission file in the required format.
    """
    model.eval()
    results = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            rec_ids = batch["rec_id"].numpy()

            # TTA: Standard + Horizontal Flip
            # 1. Forward pass original
            out1 = model(images)

            # 2. Forward pass flipped
            images_flipped = torch.flip(images, [3])  # [B, C, H, W], flip width
            out2 = model(images_flipped)

            # Average logits
            outputs = (out1 + out2) / 2.0
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Format results
            for i in range(len(rec_ids)):
                rid = rec_ids[i]
                p = probs[i]
                for species_idx in range(Config.NUM_CLASSES):
                    # Id format: rec_id * 100 + species_number
                    row_id = int(rid * 100 + species_idx)
                    probability = p[species_idx]
                    results.append({"Id": row_id, "Probability": probability})

    df = pd.DataFrame(results)
    # Sort by Id to match sample submission structure
    df = df.sort_values("Id")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
