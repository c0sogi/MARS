import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from library.config import Config
from library.utils import laplace_log_likelihood, create_submission
from library.model import VaryingCoeffNet


def train_fvc_epoch(model, loader, optimizer, device):
    """
    Trains the FVC trajectory head (Phase 1).
    Optimizes L1 Loss between predicted FVC and target FVC.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        imgs, tabs, weeks, targets = [x.to(device) for x in batch]

        optimizer.zero_grad()
        fvc_pred, _ = model(imgs, tabs, weeks)

        # L1 Loss for FVC (optimizing for median/Laplace location)
        loss = nn.functional.l1_loss(fvc_pred.squeeze(), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def train_uncertainty_epoch(model, loader, optimizer, device):
    """
    Trains the Uncertainty head (Phase 2).
    Optimizes L1 Loss between predicted Delta and absolute residuals from the frozen trajectory head.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        imgs, tabs, weeks, targets = [x.to(device) for x in batch]

        optimizer.zero_grad()

        # Get residuals from frozen trajectory head
        # We use no_grad to ensure no computation graph is built for the trajectory part
        with torch.no_grad():
            fvc_pred, _ = model(imgs, tabs, weeks)
            residuals = torch.abs(targets - fvc_pred.squeeze())

        # Predict Delta (Uncertainty)
        # Gradients will only flow through the uncertainty head
        _, delta_pred = model(imgs, tabs, weeks)

        # Loss: L1 between predicted Delta and actual Residual
        loss = nn.functional.l1_loss(delta_pred.squeeze(), residuals)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def evaluate_model(model, loader, device, metric="mae"):
    """
    Evaluates the model.
    metric="mae": Returns Mean Absolute Error of FVC predictions (for Phase 1).
    metric="laplace": Returns the modified Laplace Log Likelihood (for Phase 2).
    """
    model.eval()

    if metric == "mae":
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                imgs, tabs, weeks, targets = [x.to(device) for x in batch]
                fvc_pred, _ = model(imgs, tabs, weeks)
                loss = nn.functional.l1_loss(fvc_pred.squeeze(), targets)
                total_loss += loss.item() * imgs.size(0)
        return total_loss / len(loader.dataset)

    elif metric == "laplace":
        all_true = []
        all_pred = []
        all_sigma = []

        with torch.no_grad():
            for batch in loader:
                imgs, tabs, weeks, targets = [x.to(device) for x in batch]
                fvc_pred, delta_pred = model(imgs, tabs, weeks)

                # Analytical Scaling: Sigma = Delta * sqrt(2)
                sigma_pred = delta_pred * np.sqrt(2)

                all_true.extend(targets.cpu().numpy())
                all_pred.extend(fvc_pred.cpu().numpy().flatten())
                all_sigma.extend(sigma_pred.cpu().numpy().flatten())

        return laplace_log_likelihood(all_true, all_pred, all_sigma)

    return 0.0


def predict(model, loader, device):
    """
    Generates inference outputs for the test set.
    Returns: ids, fvc_predictions, confidence_predictions
    """
    model.eval()
    all_ids = []
    all_fvc = []
    all_conf = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns: img, tab, weeks, ids
            imgs, tabs, weeks, ids = batch
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            weeks = weeks.to(device)

            fvc_pred, delta_pred = model(imgs, tabs, weeks)

            # Analytical Scaling: Sigma = Delta * sqrt(2)
            sigma_pred = delta_pred * np.sqrt(2)

            all_ids.extend(ids)
            all_fvc.extend(fvc_pred.cpu().numpy().flatten())
            all_conf.extend(sigma_pred.cpu().numpy().flatten())

    return all_ids, all_fvc, all_conf


class Engine:
    """
    Orchestrates the training and submission process.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device

    def run_training(self, train_loader, val_loader):
        """
        Executes the two-phase training strategy.
        """
        # Initialize Model
        # tab_dim=8 derived from: 3 continuous + 2 sex (one-hot) + 3 smoking (one-hot)
        model = VaryingCoeffNet(tab_dim=8)
        model = model.to(self.device)

        print(f"Starting training on {self.device}...")

        # ==========================
        # Phase 1: FVC Trajectory
        # ==========================
        print("\n=== Phase 1: Training FVC Trajectory ===")

        # Freeze Uncertainty Head, Unfreeze Trajectory & Pooling
        for param in model.unc_mlp.parameters():
            param.requires_grad = False
        for param in model.traj_mlp.parameters():
            param.requires_grad = True
        for param in model.pool.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = train_fvc_epoch(model, train_loader, optimizer, self.device)
            val_loss = evaluate_model(model, val_loader, self.device, metric="mae")

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered for Phase 1.")
                    break

        # Load best FVC model state before starting Phase 2
        model.load_state_dict(best_state)

        # ==========================
        # Phase 2: Uncertainty
        # ==========================
        print("\n=== Phase 2: Training Uncertainty Head ===")

        # Freeze Trajectory & Pooling, Unfreeze Uncertainty Head
        for param in model.traj_mlp.parameters():
            param.requires_grad = False
        for param in model.pool.parameters():
            param.requires_grad = False
        for param in model.unc_mlp.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Maximize Laplace Metric
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_metric = -float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = train_uncertainty_epoch(
                model, train_loader, optimizer, self.device
            )
            val_metric = evaluate_model(
                model, val_loader, self.device, metric="laplace"
            )

            scheduler.step(val_metric)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Residual MAE: {train_loss:.4f} | Val Metric: {val_metric}"
            )

            if val_metric > best_metric:
                best_metric = val_metric
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered for Phase 2.")
                    break

        model.load_state_dict(best_state)
        print(f"Training Complete. Best Validation Metric: {best_metric}")
        return model

    def generate_submission(self, model, test_loader):
        """
        Generates predictions for the test set and saves them to the submission file.
        """
        print("Generating submission...")
        ids, fvc, conf = predict(model, test_loader, self.device)
        create_submission(ids, fvc, conf, Config.SUBMISSION_PATH)
