import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from library.utils import AverageMeter
from library.dataset import mixup_criterion


def train_one_epoch(loader, model, criterion, optimizer, device, mixup_alpha):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs: inputs is ((B, C, H, W), (B,))
        imgs, metas = inputs

        imgs = imgs.to(device)
        metas = metas.to(device)
        targets = targets.to(device)

        # Apply Mixup
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = imgs.size(0)
            index = torch.randperm(batch_size).to(device)

            mixed_imgs = lam * imgs + (1 - lam) * imgs[index, :]
            mixed_metas = lam * metas + (1 - lam) * metas[index]

            y_a, y_b = targets, targets[index]

            # Forward pass with mixed inputs
            outputs = model((mixed_imgs, mixed_metas))
            # Squeeze outputs to match target shape if necessary (B, 1) -> (B)
            outputs = outputs.squeeze(1)

            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model((imgs, metas))
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            imgs, metas = inputs
            imgs = imgs.to(device)
            metas = metas.to(device)
            targets = targets.to(device)

            outputs = model((imgs, metas))
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), imgs.size(0))

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    # Calculate ROC AUC
    # Handle edge case if only one class is present in batch (unlikely for full val set)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return losses.avg, auc


class Trainer:
    """
    Manages the training lifecycle, including SWA and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE

        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Standard Scheduler (Cosine Annealing)
        # Used until SWA starts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.SWA_START_EPOCH, eta_min=1e-5
        )

        # SWA Components
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.SWA_LR)

        self.best_auc = 0.0

    def fit(self):
        print(f"Starting training on {self.device}...")

        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss = train_one_epoch(
                self.train_loader,
                self.model,
                self.criterion,
                self.optimizer,
                self.device,
                self.config.MIXUP_ALPHA,
            )

            # Validation (using the current weights, not SWA yet)
            val_loss, val_auc = validate(
                self.val_loader, self.model, self.criterion, self.device
            )

            # Update SWA or Standard Scheduler
            if epoch > self.config.SWA_START_EPOCH:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                lr = self.swa_scheduler.get_last_lr()[0]
                mode = "SWA"
            else:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                mode = "STD"

            # Logging
            print(
                f"Epoch {epoch}/{self.config.EPOCHS} [{mode}] - "
                f"LR: {lr:.6f} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val AUC: {val_auc:.10f}"
            )

            # Save Best Model (Standard)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                print(f"  -> New Best Model Saved! AUC: {val_auc:.10f}")

        print("\nTraining complete. Finalizing SWA Model...")

        # Update BN statistics for SWA model
        # update_bn expects the loader to yield input samples.
        # Our loader yields ((img, meta), label).
        # update_bn iterates: for input, _ in loader: model(input)
        # This matches our structure perfectly.
        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Save SWA Model
        torch.save(self.swa_model.state_dict(), self.config.FINAL_SWA_MODEL_PATH)
        print(f"SWA Model saved to {self.config.FINAL_SWA_MODEL_PATH}")
