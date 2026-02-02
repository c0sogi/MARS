import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import AverageMeter, save_checkpoint


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, mode="max", delta=0.0001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            mode (str): 'min' for loss, 'max' for metrics like AUC.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.val_score = score
        elif self.check_improvement(score):
            self.best_score = score
            self.val_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def check_improvement(self, score):
        if self.mode == "max":
            return score > (self.best_score + self.delta)
        else:
            return score < (self.best_score - self.delta)


class SWAManager:
    """
    Manages Stochastic Weight Averaging (SWA) model, scheduler, and updates.
    """

    def __init__(self, model, optimizer, swa_start_epoch, swa_lr, device):
        self.swa_model = AveragedModel(model).to(device)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        self.swa_start_epoch = swa_start_epoch
        self.device = device
        self.start_epoch_reached = False

    def step(self, epoch, model):
        """
        Updates SWA model parameters and scheduler if start epoch is reached.
        """
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
            self.start_epoch_reached = True

    def update_bn(self, loader):
        """
        Updates BatchNorm statistics for the SWA model.
        """
        if self.start_epoch_reached:
            print("Updating SWA BatchNorm statistics...")
            self.swa_model.train()  # BN update requires train mode
            update_bn(loader, self.swa_model, device=self.device)


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y


def train_one_epoch(
    model, loader, optimizer, device, epoch, scheduler=None, use_mixup=True
):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # BCEWithLogitsLoss is standard for multi-label
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup:
            images, labels = mixup_data(
                images, labels, alpha=Config.MIXUP_ALPHA, device=device
            )

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step scheduler if it's per-iteration (optional, usually per epoch for this task)
        # if scheduler: scheduler.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch}: Train Loss: {losses.avg}")
    return losses.avg


def valid_one_epoch(model, loader, device, epoch):
    """
    Validates the model for one epoch.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute ROC AUC (Macro Average) manually to handle degenerate classes
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    class_aucs = []
    for i in range(Config.NUM_CLASSES):
        # Check if the class has both positive and negative samples
        if len(np.unique(all_targets[:, i])) > 1:
            class_auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            class_aucs.append(class_auc)

    if len(class_aucs) > 0:
        auc = np.mean(class_aucs)
    else:
        auc = 0.0
        print("Warning: No valid classes for AUC calculation.")

    print(f"Epoch {epoch}: Val Loss: {losses.avg}, Val AUC: {auc}")
    return losses.avg, auc


def inference_fn(model, loader, device, use_tta=True):
    """
    Generates predictions for the test set, optionally using TTA.
    """
    model.eval()
    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)

            # Standard Forward Pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Horizontal Flip TTA
                # Input is (B, C, H, W). Flip on dim 3 (Width/Time).
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs = (probs + probs_flip) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_rec_ids.append(rec_ids.numpy())

    return np.concatenate(all_probs), np.concatenate(all_rec_ids)


def generate_submission(probs, rec_ids, output_path):
    """
    Formats the predictions and saves the submission CSV.

    Args:
        probs (np.array): Shape (N, 19) - Predicted probabilities.
        rec_ids (np.array): Shape (N,) - Recording IDs.
        output_path (str): Path to save the CSV.
    """
    submission_rows = []

    # Iterate through each recording
    for i in range(len(rec_ids)):
        r_id = int(rec_ids[i])
        p_vec = probs[i]

        # For each species (0-18)
        for species_idx in range(Config.NUM_CLASSES):
            # Construct the combined Id: rec_id * 100 + species_id
            row_id = r_id * 100 + species_idx
            probability = p_vec[species_idx]

            submission_rows.append({"Id": row_id, "Probability": probability})

    df_sub = pd.DataFrame(submission_rows)

    # Ensure integer Id
    df_sub["Id"] = df_sub["Id"].astype(int)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
