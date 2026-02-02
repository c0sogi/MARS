import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
import numpy as np
from library import config, utils


class Engine:
    """
    Handles training and validation logic for the Iceberg Classification task.
    """

    def __init__(self, model, device, optimizer=None, criterion=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = utils.get_logger("Engine")

    def train_one_epoch(self, loader, epoch):
        """
        Trains the model for one epoch using the SAM optimizer.
        """
        self.model.train()
        losses = utils.AverageMeter()
        accuracies = utils.AverageMeter()

        for batch_idx, (images, angles, targets) in enumerate(loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            # --- SAM Step 1: Compute gradients for current weights ---
            # Forward pass
            logits = self.model(images, angles)
            loss = self.criterion(logits, targets)

            # Backward pass to populate gradients
            loss.backward()

            # --- SAM Step 2: Perturb weights and update ---
            # Define closure for the second forward-backward pass at perturbed state
            def closure():
                self.optimizer.zero_grad()
                logits_adv = self.model(images, angles)
                loss_adv = self.criterion(logits_adv, targets)
                loss_adv.backward()
                return loss_adv

            # Optimizer step (applies perturbation -> closure -> restores -> updates)
            self.optimizer.step(closure)
            self.optimizer.zero_grad()

            # --- Metrics ---
            # Apply sigmoid for accuracy calculation
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == targets).float().mean()

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))

        return losses.avg, accuracies.avg

    def validate_tta(self, loader):
        """
        Validates the model using Test-Time Augmentation (Klein Four-Group).
        TTA: Original, H-Flip, V-Flip, Rot180.
        """
        self.model.eval()
        losses = utils.AverageMeter()
        accuracies = utils.AverageMeter()

        # We use BCELoss for probabilities since we average probabilities from TTA
        # Note: The training criterion is BCEWithLogitsLoss (takes logits),
        # but here we manually apply sigmoid and average, so we compare probs.
        val_criterion = nn.BCELoss()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, angles, targets in loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                # 1. Original
                logits_1 = self.model(images, angles)
                probs_1 = torch.sigmoid(logits_1)

                # 2. Horizontal Flip (dim 3)
                images_h = torch.flip(images, [3])
                logits_2 = self.model(images_h, angles)
                probs_2 = torch.sigmoid(logits_2)

                # 3. Vertical Flip (dim 2)
                images_v = torch.flip(images, [2])
                logits_3 = self.model(images_v, angles)
                probs_3 = torch.sigmoid(logits_3)

                # 4. Rotate 180 (H + V flip) (dim 2, 3)
                images_r = torch.flip(images, [2, 3])
                logits_4 = self.model(images_r, angles)
                probs_4 = torch.sigmoid(logits_4)

                # Average Probabilities
                avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

                # Calculate Loss
                loss = val_criterion(avg_probs, targets)

                # Calculate Accuracy
                preds = (avg_probs > 0.5).float()
                acc = (preds == targets).float().mean()

                losses.update(loss.item(), images.size(0))
                accuracies.update(acc.item(), images.size(0))

                all_preds.extend(avg_probs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        return losses.avg, accuracies.avg, np.array(all_preds), np.array(all_targets)

    def predict_test_tta(self, loader):
        """
        Generates predictions for the test set using TTA.
        """
        self.model.eval()
        all_ids = []
        all_probs = []

        with torch.no_grad():
            for images, angles, ids in loader:
                images = images.to(self.device)
                angles = angles.to(self.device)

                # TTA
                # 1. Original
                p1 = torch.sigmoid(self.model(images, angles))
                # 2. H-Flip
                p2 = torch.sigmoid(self.model(torch.flip(images, [3]), angles))
                # 3. V-Flip
                p3 = torch.sigmoid(self.model(torch.flip(images, [2]), angles))
                # 4. Rot180
                p4 = torch.sigmoid(self.model(torch.flip(images, [2, 3]), angles))

                avg_probs = (p1 + p2 + p3 + p4) / 4.0

                all_probs.extend(avg_probs.cpu().numpy().flatten())
                all_ids.extend(ids.numpy() if isinstance(ids, torch.Tensor) else ids)

        return all_ids, np.array(all_probs)


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    Wraps the AveragedModel and handles custom BatchNorm updates for the dual-input model.
    """

    def __init__(self, model, device):
        self.device = device
        # Create SWA model wrapper
        self.swa_model = AveragedModel(model).to(device)

    def update(self, model):
        """
        Updates the SWA model parameters with the current model's parameters.
        """
        self.swa_model.update_parameters(model)

    def update_bn(self, loader):
        """
        Custom implementation of update_bn for models with multiple inputs (image + angle).
        Standard torch.optim.swa_utils.update_bn assumes model(x) signature.
        """
        # Reset BatchNorm statistics
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                module.momentum = None  # Use simple average during this pass

        self.swa_model.train()  # Set to train mode to update stats

        with torch.no_grad():
            for images, angles, _ in loader:
                images = images.to(self.device)
                angles = angles.to(self.device)

                # Forward pass updates the running statistics
                self.swa_model(images, angles)

    def get_model(self):
        return self.swa_model
