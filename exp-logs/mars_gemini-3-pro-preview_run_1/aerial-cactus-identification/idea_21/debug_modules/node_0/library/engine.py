import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR

from library.config import DEVICE, MIXUP_ALPHA
from library.utils import AverageMeter, calculate_roc_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform operations on.
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


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """
    Computes the mixup loss summing over all model heads.
    Args:
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        preds (list): List of output tensors from the model's heads.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixup lambda.
    """
    total_loss = 0
    # preds is a list of [head1_out, head2_out, head3_out]
    for pred in preds:
        # Flatten pred to match target shape (Batch_Size,)
        pred_flat = pred.view(-1)
        loss = lam * criterion(pred_flat, y_a) + (1 - lam) * criterion(pred_flat, y_b)
        total_loss += loss
    return total_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Head Loss.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, MIXUP_ALPHA, device
        )

        # Forward pass
        # Model returns list: [out1, out2, out3]
        outputs = model(mixed_images)

        # Compute Loss (Sum of all heads)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model.
    Aggregates predictions from all 3 heads.
    Can handle dataloaders with or without labels (validation vs inference).
    """
    model.eval()

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Handle both (image, label) and (image) batches
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
                labels = labels.to(device)
            else:
                images = batch
                labels = None

            images = images.to(device)

            # Forward pass
            outputs = model(images)  # [out1, out2, out3]

            # Compute Loss if targets available
            if labels is not None:
                # Standard BCE for validation (no mixup)
                batch_loss = 0
                for out in outputs:
                    batch_loss += criterion(out.view(-1), labels)
                losses.update(batch_loss.item(), images.size(0))

                all_targets.extend(labels.cpu().numpy())

            # Aggregate predictions (Average of Sigmoids)
            # out shape: (B, 1)
            probs_list = [torch.sigmoid(out).view(-1) for out in outputs]
            # Stack to (3, B) then mean to (B,)
            avg_probs = torch.stack(probs_list).mean(dim=0)

            all_preds.extend(avg_probs.cpu().numpy())

    # Calculate Metric
    auc = 0.0
    if len(all_targets) > 0:
        auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
        print(f"Validation Loss: {losses.avg}")
        print(f"Validation ROC AUC: {auc}")
    else:
        print("Inference complete (no labels).")

    return losses.avg, auc, np.array(all_preds)


def make_submission(model, test_loader, test_ids, output_path, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print(f"Generating submission for {len(test_ids)} samples...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Aggregate predictions (Average of Sigmoids)
            probs_list = [torch.sigmoid(out).view(-1) for out in outputs]
            avg_probs = torch.stack(probs_list).mean(dim=0)

            all_preds.extend(avg_probs.cpu().numpy())

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "has_cactus": all_preds})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    """

    def __init__(self, model, optimizer, swa_lr):
        self.model = model  # Reference to the training model
        self.optimizer = optimizer
        self.swa_model = AveragedModel(model).to(DEVICE)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    def update(self):
        """
        Updates the SWA model with current weights and steps the SWA scheduler.
        Should be called at the end of each SWA epoch.
        """
        self.swa_model.update_parameters(self.model)
        self.swa_scheduler.step()

    def update_bn(self, dataloader, device):
        """
        Updates BatchNorm statistics for the SWA model.
        """
        print("Updating SWA BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(dataloader, self.swa_model, device=device)

    def get_averaged_model(self):
        """
        Returns the averaged model.
        """
        return self.swa_model
