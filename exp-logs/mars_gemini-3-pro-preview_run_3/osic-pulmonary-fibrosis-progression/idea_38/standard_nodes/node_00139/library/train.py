import os
import torch
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import (
    seed_everything,
    laplace_log_likelihood,
    metric_aligned_nll_loss,
)
from library.data import get_dataloaders
from library.model import DSPRNet


def train_one_epoch(model, loader, optimizer, device, fvc_mean, fvc_std):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, clinical, linear_input, targets, _ in loader:
        images = images.to(device)
        clinical = clinical.to(device)
        linear_input = linear_input.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        mu_final, sigma_final = model(images, clinical, linear_input)

        # Denormalize for physical unit loss calculation
        targets_dn = targets * fvc_std + fvc_mean
        mu_final_dn = mu_final * fvc_std + fvc_mean
        sigma_final_dn = sigma_final * fvc_std

        # Cite solution_lesson_node_00138: Use smooth surrogate loss.
        loss = metric_aligned_nll_loss(targets_dn, mu_final_dn, sigma_final_dn)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device, fvc_mean, fvc_std):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_metric = 0.0

    with torch.no_grad():
        for images, clinical, linear_input, targets, _ in loader:
            images = images.to(device)
            clinical = clinical.to(device)
            linear_input = linear_input.to(device)
            targets = targets.to(device)

            # Forward pass
            mu_final, sigma_final = model(images, clinical, linear_input)

            # Denormalize
            targets_dn = targets * fvc_std + fvc_mean
            mu_final_dn = mu_final * fvc_std + fvc_mean
            sigma_final_dn = sigma_final * fvc_std

            # Calculate Metric (with clipping for evaluation)
            batch_metric = laplace_log_likelihood(
                targets_dn, mu_final_dn, sigma_final_dn
            )
            total_metric += batch_metric.item() * images.size(0)

    avg_metric = total_metric / len(loader.dataset)
    return avg_metric


def run_inference(test_loader, device):
    """
    Loads the best model and generates the submission file.
    """
    model = DSPRNet().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using initialized model.")

    model.eval()
    results = []

    stats = test_loader.dataset.stats
    fvc_mean = stats.get("fvc_mean", 2500.0)
    fvc_std = stats.get("fvc_std", 500.0)

    with torch.no_grad():
        for images, clinical, linear_input, _, meta in test_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            linear_input = linear_input.to(device)

            mu_final, sigma_final = model(images, clinical, linear_input)

            mu_np = mu_final.cpu().numpy().flatten()
            sigma_np = sigma_final.cpu().numpy().flatten()

            # Inverse Transform
            pred_fvc = mu_np * fvc_std + fvc_mean
            pred_sigma = sigma_np * fvc_std

            # Extract metadata
            # meta is a dict where values are lists/tensors
            pat_weeks = meta["Patient_Week"]

            for i in range(len(pat_weeks)):
                # Post-processing clip for submission
                conf = max(pred_sigma[i], 70)

                results.append(
                    {
                        "Patient_Week": pat_weeks[i],
                        "FVC": pred_fvc[i],
                        "Confidence": conf,
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main execution function.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()

    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Normalization stats for loss calculation
    stats = train_loader.dataset.stats
    fvc_mean = stats.get("fvc_mean", 2500.0)
    fvc_std = stats.get("fvc_std", 500.0)

    # Model Initialization
    model = DSPRNet().to(device)

    # Optimizer with differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_metric = -float("inf")
    patience = 10  # Early stopping patience
    counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, fvc_mean, fvc_std
        )
        val_metric = validate(model, val_loader, device, fvc_mean, fvc_std)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved! Metric: {val_metric}")
            counter = 0
        else:
            counter += 1

        # Early Stopping
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Run Inference
    print("Running inference with best model...")
    run_inference(test_loader, device)
