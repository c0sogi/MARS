import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

from library.config import Config
from library.utils import calculate_roc_auc
from library.data import mixup_data


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, file_sizes, labels) in enumerate(loader):
        images = images.to(device)
        file_sizes = file_sizes.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Apply Mixup if enabled
        if Config.MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images, file_sizes)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )
        else:
            outputs = model(images, file_sizes)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, file_sizes, labels in loader:
            images = images.to(device)
            file_sizes = file_sizes.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, file_sizes)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    auc_score = calculate_roc_auc(all_labels, all_preds)

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation AUC: {auc_score}")

    return epoch_loss, auc_score


class SWAHandler:
    """
    Handles Stochastic Weight Averaging (SWA) logic.
    """

    def __init__(self, model, optimizer, swa_start_epoch, swa_lr, device):
        self.swa_model = AveragedModel(model).to(device)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        self.swa_start_epoch = swa_start_epoch
        self.device = device
        self.active = False

    def step(self, epoch, model):
        """
        Updates SWA model parameters and scheduler if the start epoch is reached.
        Returns True if SWA step was performed.
        """
        if epoch >= self.swa_start_epoch:
            self.active = True
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
            return True
        return False

    def update_bn(self, loader):
        """
        Updates Batch Normalization statistics for the SWA model.
        Custom loop required because the model takes (image, metadata).
        """
        print("Updating SWA Batch Normalization statistics...")
        self.swa_model.train()
        with torch.no_grad():
            for images, file_sizes, _ in loader:
                images = images.to(self.device)
                file_sizes = file_sizes.to(self.device)
                # Forward pass to update running mean/var
                _ = self.swa_model(images, file_sizes)

    def get_model(self):
        return self.swa_model


def inference_tta(model, loader, device):
    """
    Performs inference using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, 180 Rotation.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, file_sizes in loader:
            images = images.to(device)
            file_sizes = file_sizes.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(images, file_sizes))

            # 2. Horizontal Flip
            img_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(img_h, file_sizes))

            # 3. Vertical Flip
            img_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(img_v, file_sizes))

            # 4. Rotate 180 (equivalent to H-flip + V-flip)
            img_r = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(img_r, file_sizes))

            # Average predictions
            avg_pred = (out1 + out2 + out3 + out4) / 4.0
            all_preds.append(avg_pred.cpu().numpy())

    return np.concatenate(all_preds)


class EarlyStopping:
    """
    Early stopping to stop training when the metric has stopped improving.
    """

    def __init__(self, patience=7, mode="max", delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta
        if mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, score):
        """
        Returns True if the current model should be saved (improvement found).
        Sets self.early_stop to True if patience is exceeded.
        """
        if self.mode == "min":
            improved = (
                score < (self.best_score - self.delta)
                if self.best_score is not None
                else True
            )
        else:
            improved = (
                score > (self.best_score + self.delta)
                if self.best_score is not None
                else True
            )

        if self.best_score is None:
            self.best_score = score
            return True  # Save checkpoint
        elif improved:
            self.best_score = score
            self.counter = 0
            return True  # Save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
