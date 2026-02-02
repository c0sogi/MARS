import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import (
    mixup_data,
    mixup_criterion,
    compute_robust_auc,
    save_checkpoint,
)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and BCEWithLogitsLoss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (inputs, targets, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        outputs = model(inputs)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and robust AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for AUC calculation
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc = compute_robust_auc(all_targets, all_preds)

    return avg_loss, auc


class CheckpointTracker:
    """
    Manages the saving and deletion of checkpoints to maintain the Top-K best models.
    """

    def __init__(self, fold, model_name, save_dir, top_k=Config.TOP_K_CHECKPOINTS):
        self.fold = fold
        self.model_name = model_name
        self.save_dir = save_dir
        self.top_k = top_k
        # List of tuples: (score, file_path)
        self.best_checkpoints = []

    def update(self, model, optimizer, epoch, score):
        """
        Updates the tracker with a new checkpoint candidate.
        If the score is good enough, saves the checkpoint and manages the top-k list.
        """
        filename = f"{self.model_name}_fold{self.fold}_ep{epoch}.pth"
        filepath = os.path.join(self.save_dir, filename)

        # Determine if we should save this checkpoint
        should_save = False
        if len(self.best_checkpoints) < self.top_k:
            should_save = True
        else:
            # Check if better than the worst in the top_k
            # Assuming higher score is better (AUC)
            min_score = self.best_checkpoints[-1][0]
            if score > min_score:
                should_save = True

        if should_save:
            # Save the new checkpoint
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": float(score),
            }
            save_checkpoint(state, filepath)

            # Add to list and sort
            self.best_checkpoints.append((score, filepath))
            # Sort descending by score
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Prune if exceeding top_k
            while len(self.best_checkpoints) > self.top_k:
                to_remove = self.best_checkpoints.pop()
                remove_path = to_remove[1]
                if os.path.exists(remove_path):
                    try:
                        os.remove(remove_path)
                    except OSError:
                        pass

        return should_save


def fit_model(
    model, train_loader, val_loader, optimizer, scheduler, device, fold, model_name
):
    """
    Runs the full training loop with Early Stopping and Top-K Checkpointing.
    """
    tracker = CheckpointTracker(
        fold=fold, model_name=model_name, save_dir=Config.CHECKPOINT_DIR
    )

    criterion = nn.BCEWithLogitsLoss()

    best_score_overall = -float("inf")
    patience_counter = 0

    print(f"Starting training for {model_name} - Fold {fold}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Update Checkpoints
        tracker.update(model, optimizer, epoch + 1, val_auc)

        # Early Stopping Logic
        if val_auc > best_score_overall:
            best_score_overall = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return tracker.best_checkpoints
