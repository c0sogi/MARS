import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import calculate_auc


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for Mixup training.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_fn):
    """
    Trains the model for one epoch using Mixup and Dual-Head Loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        # Apply Mixup
        images, y_a, y_b, lam = mixup_fn(images, labels)

        optimizer.zero_grad()

        # Forward pass (returns two logits: Texture and Semantic)
        logits_texture, logits_semantic = model(images)

        # Calculate loss for both heads
        loss_texture = mixup_criterion(criterion, logits_texture, y_a, y_b, lam)
        loss_semantic = mixup_criterion(criterion, logits_semantic, y_a, y_b, lam)

        # Total loss is the sum of both heads
        loss = loss_texture + loss_semantic

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

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            logits_texture, logits_semantic = model(images)

            # Loss calculation (Standard BCE, sum of heads)
            loss_texture = criterion(logits_texture, labels)
            loss_semantic = criterion(logits_semantic, labels)
            loss = loss_texture + loss_semantic

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Predictions: Sigmoid and Average of both heads
            p_texture = torch.sigmoid(logits_texture)
            p_semantic = torch.sigmoid(logits_semantic)
            avg_p = (p_texture + p_semantic) / 2.0

            preds_list.append(avg_p.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(preds_list)
    all_targets = np.concatenate(targets_list)

    auc_score = calculate_auc(all_targets, all_preds)

    return avg_loss, auc_score


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    """

    def __init__(self, model, device, swa_start_epoch):
        self.swa_model = AveragedModel(model).to(device)
        self.swa_start_epoch = swa_start_epoch
        self.device = device

    def update(self, model, epoch):
        """
        Updates the SWA model parameters if the current epoch is past the start epoch.
        """
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(model)

    def update_bn(self, loader):
        """
        Updates Batch Normalization statistics for the SWA model.
        """
        # update_bn expects the loader to yield batches.
        # It handles the forward pass internally to update running stats.
        update_bn(loader, self.swa_model, device=self.device)

    def get_model(self):
        return self.swa_model


def predict_with_tta(model, images):
    """
    Helper to predict a batch of images with Test Time Augmentation (4 views).
    Returns tensor of shape (B, 1) with probabilities.
    """
    # images: (B, C, H, W)
    batch_size = images.size(0)

    # Define augmentations: Original, HFlip, VFlip, Rot180
    views = [
        images,
        torch.flip(images, [3]),  # HFlip
        torch.flip(images, [2]),  # VFlip
        torch.flip(images, [2, 3]),  # Rot180 (HFlip + VFlip)
    ]

    total_probs = torch.zeros(batch_size, 1, device=images.device)

    for view in views:
        # Forward pass on the view
        l_tex, l_sem = model(view)

        # Sigmoid
        p_tex = torch.sigmoid(l_tex)
        p_sem = torch.sigmoid(l_sem)

        # Average heads for this view
        p_view = (p_tex + p_sem) / 2.0
        total_probs += p_view

    # Average across all 4 views
    return total_probs / 4.0


def generate_submission(models, test_loader, test_ids, device, output_path):
    """
    Generates predictions using an ensemble of models (folds) with TTA.
    Saves the result to CSV.
    """
    print(f"Generating submission with ensemble of {len(models)} models...")

    # Ensure models are in eval mode
    for m in models:
        m.eval()

    all_probs = []

    with torch.no_grad():
        # test_loader yields only images as per CactusDataset implementation
        for images in test_loader:
            images = images.to(device)

            # Accumulator for the batch
            batch_probs = torch.zeros(images.size(0), 1, device=device)

            for model in models:
                # Predict with TTA for this model
                p = predict_with_tta(model, images)
                batch_probs += p

            # Average across models (Folds)
            batch_probs /= len(models)
            all_probs.append(batch_probs.cpu().numpy())

    # Flatten results
    final_probs = np.concatenate(all_probs).flatten()

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
