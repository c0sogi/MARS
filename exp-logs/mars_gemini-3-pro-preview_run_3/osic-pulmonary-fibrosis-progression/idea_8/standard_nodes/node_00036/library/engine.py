import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import (
    STATS,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
    UNCERTAINTY_FLOOR,
    CHECKPOINT_DIR,
)
from library.utils import LaplaceLogLikelihood


class TAPNetEngine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Stats for unscaling during validation
        self.fvc_mean = STATS["FVC_MEAN"]
        self.fvc_std = STATS["FVC_STD"]

    def calculate_trajectory(self, params, tabular, weeks):
        """
        Computes mu(t) and sigma(t) based on model parameters and inputs.

        params: (B, 5) -> [alpha, beta, gamma, delta_base, delta_growth]
        tabular: (B, 7) -> [Base_FVC, ...] (Index 0 is Base_FVC)
        weeks: (B, 1) -> Scaled weeks
        """
        # Extract parameters
        alpha = params[:, 0:1]
        beta = params[:, 1:2]
        gamma = params[:, 2:3]
        delta_base = params[:, 3:4]
        delta_growth = params[:, 4:5]

        # Extract Baseline FVC (scaled) from tabular input
        baseline_fvc = tabular[:, 0:1]

        # 1. Mean Trajectory: mu(t)
        # mu(t) = alpha * Baseline + beta + gamma * Weeks
        mu = alpha * baseline_fvc + beta + gamma * weeks

        # 2. Uncertainty Trajectory: sigma(t)
        # sigma(t) = softplus(delta_base) + softplus(delta_growth) * |Weeks| + epsilon
        # We take absolute value of weeks because uncertainty grows in both past and future directions
        sigma = (
            F.softplus(delta_base)
            + F.softplus(delta_growth) * torch.abs(weeks)
            + UNCERTAINTY_FLOOR
        )

        return mu, sigma

    def criterion(self, params, tabular, weeks, target_fvc):
        """
        Computes the Laplace Negative Log Likelihood loss.
        Cite solution_lesson_node_00034: Avoid Auxiliary Objectives that Conflict with Probabilistic Targets.
        """
        # Calculate trajectory in scaled space
        mu, sigma = self.calculate_trajectory(params, tabular, weeks)

        # Primary Loss: Laplace Negative Log Likelihood
        # Metric definition: Score = - (sqrt(2) * Delta) / sigma - ln(sqrt(2) * sigma)
        # NLL (to minimize) = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)

        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=self.device))
        absolute_error = torch.abs(target_fvc - mu)

        nll = (sqrt_2 * absolute_error) / sigma + torch.log(sqrt_2 * sigma)
        nll_loss = torch.mean(nll)

        return nll_loss

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Move data to device
            image = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            weeks = batch["weeks"].to(self.device)
            target_fvc = batch["target_fvc"].to(self.device)

            # Zero gradients
            if self.optimizer:
                self.optimizer.zero_grad()

            # Forward pass
            params = self.model(image, tabular)

            # Compute loss
            loss = self.criterion(params, tabular, weeks, target_fvc)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.optimizer:
                self.optimizer.step()

            # Update stats
            batch_size = image.size(0)
            running_loss += loss.item() * batch_size

        if self.scheduler:
            self.scheduler.step()

        dataset_size = len(dataloader.dataset)
        epoch_loss = running_loss / dataset_size

        return {"loss": epoch_loss}

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_true_fvc = []
        all_pred_fvc = []
        all_pred_sigma = []

        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                weeks = batch["weeks"].to(self.device)
                target_fvc = batch["target_fvc"].to(self.device)

                # Forward pass
                params = self.model(image, tabular)

                # Compute Validation Loss (Scaled NLL)
                loss = self.criterion(params, tabular, weeks, target_fvc)
                batch_size = image.size(0)
                running_loss += loss.item() * batch_size

                # Compute Trajectory (Scaled)
                mu_scaled, sigma_scaled = self.calculate_trajectory(
                    params, tabular, weeks
                )

                # Unscale for Metric Calculation
                # mu_raw = mu_scaled * std + mean
                mu_raw = mu_scaled * self.fvc_std + self.fvc_mean
                # sigma_raw = sigma_scaled * std
                sigma_raw = sigma_scaled * self.fvc_std
                # target_raw = target_scaled * std + mean
                target_raw = target_fvc * self.fvc_std + self.fvc_mean

                all_true_fvc.append(target_raw.cpu())
                all_pred_fvc.append(mu_raw.cpu())
                all_pred_sigma.append(sigma_raw.cpu())

        # Aggregate results
        dataset_size = len(dataloader.dataset)
        epoch_loss = running_loss / dataset_size

        y_true = torch.cat(all_true_fvc, dim=0)
        y_pred = torch.cat(all_pred_fvc, dim=0)
        sigma_pred = torch.cat(all_pred_sigma, dim=0)

        # Compute Competition Metric
        # Note: LaplaceLogLikelihood expects unscaled values
        metric_score = LaplaceLogLikelihood(y_true, y_pred, sigma_pred)

        return {"loss": epoch_loss, "score": metric_score}

    def save_checkpoint(self, filename="best_model.pth"):
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, filename="best_model.pth"):
        load_path = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f"Loaded checkpoint from {load_path}")
        else:
            print(f"No checkpoint found at {load_path}")


def get_optimizer(model):
    """
    Sets up AdamW optimizer with differential learning rates.
    """
    # Separate parameters for backbone and head
    backbone_params = []
    head_params = []

    # Identify backbone parameters (EfficientNetEncoder)
    # We assume model.image_encoder is the backbone wrapper
    backbone_ids = list(map(id, model.image_encoder.parameters()))

    for name, param in model.named_parameters():
        if id(param) in backbone_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


def get_scheduler(optimizer, epochs):
    """
    Sets up Cosine Annealing scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
