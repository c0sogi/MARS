import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.utils import MetricMonitor


def mixup_data(x, film, y_class, y_aux, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to images and FiLM features.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Mix FiLM features (metadata) linearly
    mixed_film = lam * film + (1 - lam) * film[index, :]

    # Return pairs of targets for loss calculation
    y_class_a, y_class_b = y_class, y_class[index]
    y_aux_a, y_aux_b = y_aux, y_aux[index]

    return mixed_x, mixed_film, y_class_a, y_class_b, y_aux_a, y_aux_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes the mixup loss given the criterion, predictions, and target pairs."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Multi-Task Learning and Mixup.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Loss functions
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()

    for batch_idx, (images, labels, film_feats, aux_targets) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)
        film_feats = film_feats.to(device).unsqueeze(1)  # (B, 1)
        aux_targets = aux_targets.to(device).unsqueeze(1)  # (B, 1)

        # Apply Mixup if enabled
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            images, film_feats, labels_a, labels_b, aux_a, aux_b, lam = mixup_data(
                images, film_feats, labels, aux_targets, Config.MIXUP_ALPHA, device
            )

            # Forward pass
            logits, aux_pred = model(images, film_feats)

            # Classification Loss
            loss_class = mixup_criterion(
                criterion_class, logits, labels_a, labels_b, lam
            )

            # Auxiliary Loss (Quality Regression)
            if Config.USE_MTL:
                loss_aux = mixup_criterion(criterion_aux, aux_pred, aux_a, aux_b, lam)
                loss = loss_class + Config.AUX_LOSS_WEIGHT * loss_aux
            else:
                loss = loss_class
                loss_aux = torch.tensor(0.0, device=device)

        else:
            # Standard Forward Pass
            logits, aux_pred = model(images, film_feats)
            loss_class = criterion_class(logits, labels)

            if Config.USE_MTL:
                loss_aux = criterion_aux(aux_pred, aux_targets)
                loss = loss_class + Config.AUX_LOSS_WEIGHT * loss_aux
            else:
                loss = loss_class
                loss_aux = torch.tensor(0.0, device=device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("ClassLoss", loss_class.item())
        if Config.USE_MTL:
            metric_monitor.update("AuxLoss", loss_aux.item())

    print(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.metrics["Loss"]["avg"]


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Returns ROC AUC and average loss.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    criterion_class = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels, film_feats, aux_targets in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            film_feats = film_feats.to(device)
            aux_targets = aux_targets.to(device).unsqueeze(1)

            logits, aux_pred = model(images, film_feats)

            loss_class = criterion_class(logits, labels)

            if Config.USE_MTL:
                loss_aux = criterion_aux(aux_pred, aux_targets)
                loss = loss_class + Config.AUX_LOSS_WEIGHT * loss_aux
            else:
                loss = loss_class
                loss_aux = torch.tensor(0.0, device=device)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("ClassLoss", loss_class.item())
            if Config.USE_MTL:
                metric_monitor.update("AuxLoss", loss_aux.item())

            # Store predictions for AUC calculation
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    # Calculate AUC
    # Handle edge case if batch only has one class (unlikely for full val set)
    if len(np.unique(targets)) > 1:
        auc = roc_auc_score(targets, preds)
    else:
        auc = 0.5

    print(f"Validation: {metric_monitor} | AUC: {auc}")
    return auc, metric_monitor.metrics["Loss"]["avg"]


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    Handles model updates and Batch Normalization statistics re-computation.
    """

    def __init__(self, model, device):
        self.swa_model = AveragedModel(model).to(device)
        self.device = device
        self.start_epoch = Config.SWA_START_EPOCH

    def update(self, model, epoch):
        """Updates the SWA model parameters if the start epoch is reached."""
        if epoch >= self.start_epoch:
            self.swa_model.update_parameters(model)

    def finalize(self, train_loader):
        """
        Finalizes the SWA model by updating Batch Normalization statistics.
        This is necessary because SWA averages weights, which desynchronizes BN stats.
        """
        print("SWA: Updating BN statistics...")

        # 1. Reset BN stats and set momentum to None to calculate simple average
        momenta = {}
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        # 2. Run forward passes on the training data to populate stats
        self.swa_model.train()
        with torch.no_grad():
            for images, _, film_feats, _ in train_loader:
                images = images.to(self.device)
                film_feats = film_feats.to(self.device)
                # Forward pass updates the running_mean and running_var
                self.swa_model(images, film_feats)

        # 3. Restore original momentum values
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

        return self.swa_model
