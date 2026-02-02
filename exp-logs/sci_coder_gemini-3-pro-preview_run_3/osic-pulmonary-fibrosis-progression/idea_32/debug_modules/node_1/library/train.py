import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import OSPRNet


def metric_aligned_laplace_loss(y_pred_mu, y_pred_sigma, y_true):
    """
    Computes the Metric-Aligned Laplace Log Likelihood Loss.

    Args:
        y_pred_mu (torch.Tensor): Predicted mean (scaled).
        y_pred_sigma (torch.Tensor): Predicted standard deviation (scaled).
        y_true (torch.Tensor): Ground truth target (scaled).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure sigma is positive (model uses softplus + 1e-6, but safety clamp helps)
    sigma = torch.clamp(y_pred_sigma, min=1e-6)

    # Calculate absolute error
    delta = torch.abs(y_true - y_pred_mu)

    # Constants
    sqrt_2 = np.sqrt(2)

    # Loss formula: L = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
    loss = (sqrt_2 * delta) / sigma + torch.log(sqrt_2 * sigma)

    return loss.mean()


def train_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        img, tab, target = batch
        img = img.to(device)
        tab = tab.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        mu, sigma = model(img, tab)

        loss = metric_aligned_laplace_loss(mu, sigma, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Unscales predictions to raw ml units before metric calculation.
    """
    model.eval()
    total_metric = 0.0

    with torch.no_grad():
        for batch in loader:
            img, tab, target = batch
            img = img.to(device)
            tab = tab.to(device)
            target = target.to(device)

            # Forward pass (outputs are scaled)
            mu_scaled, sigma_scaled = model(img, tab)

            # Unscale to raw ml units
            # mu_raw = mu_scaled * std + mean
            mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            # sigma_raw = sigma_scaled * std
            sigma_raw = sigma_scaled * Config.TARGET_STD

            # Unscale target
            target_raw = target * Config.TARGET_STD + Config.TARGET_MEAN

            # Compute metric (metric function handles numpy conversion and clipping)
            score = laplace_log_likelihood_metric(target_raw, mu_raw, sigma_raw)
            total_metric += score

    return total_metric / len(loader)


def generate_submission(model, loader, device):
    """
    Generates predictions for the submission set and saves to CSV.
    """
    model.eval()
    results = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in loader:
            img, tab, pw_ids = batch
            img = img.to(device)
            tab = tab.to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(img, tab)

            # Unscale
            mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma_scaled * Config.TARGET_STD

            # Convert to numpy
            mu_raw = mu_raw.cpu().numpy()
            sigma_raw = sigma_raw.cpu().numpy()

            # Apply submission constraint: Confidence must be >= 70
            # Note: The metric function does this for scoring, but we must output it explicitly
            # if we want the file to be compliant, though the metric formula usually handles it.
            # However, the task description says "confidence values are clipped at 70 ml".
            # It's safer to clip the output column.
            sigma_raw = np.maximum(sigma_raw, Config.CONFIDENCE_CLIP)

            for pw, fvc, conf in zip(pw_ids, mu_raw, sigma_raw):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    df = pd.DataFrame(results)
    # Ensure column order
    df = df[["Patient_Week", "FVC", "Confidence"]]
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data
    # load_cached_data=True allows using pre-processed .npy files if available
    train_loader, val_loader, sub_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = OSPRNet().to(Config.DEVICE)

    # 4. Optimizer with Differential Learning Rates
    # Identify backbone parameters (those that are part of the timm model and require grad)
    # We use the object identity to separate them.
    backbone_params_ids = list(map(id, model.visual_residual.backbone.parameters()))

    # Filter parameters that require gradients
    backbone_params = [
        p
        for p in model.parameters()
        if id(p) in backbone_params_ids and p.requires_grad
    ]
    head_params = [
        p
        for p in model.parameters()
        if id(p) not in backbone_params_ids and p.requires_grad
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 5. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 6. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_score = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        # Print metrics (Full precision)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Score: {val_score}")

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model (Score: {best_score})")

    print("Training complete.")

    # 7. Generate Submission
    if os.path.exists(best_model_path):
        print("Loading best model for inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: No best model found. Using current model state.")

    generate_submission(model, sub_loader, Config.DEVICE)
