import time
import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import RIDSNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: [mu, raw_sigma]
        preds = model(images, tabular)

        # Loss calculation (targets are normalized)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device, stats):
    """
    Performs validation and calculates the competition metric on unnormalized data.
    """
    model.eval()
    running_loss = 0.0

    # Lists to store unnormalized predictions for metric calculation
    all_targets = []
    all_mus = []
    all_sigmas = []

    # Statistics for inverse transformation
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            preds = model(images, tabular)

            # Compute loss in the normalized space (same as training)
            loss = criterion(preds, targets)
            running_loss += loss.item() * images.size(0)

            # Extract predictions
            mu_norm = preds[:, 0]
            raw_sigma = preds[:, 1]

            # Inverse Transform
            # 1. Sigma: Softplus to ensure positivity, then scale
            sigma_norm = F.softplus(raw_sigma)
            sigma_real = sigma_norm * fvc_std

            # 2. Mu: Unnormalize
            mu_real = mu_norm * fvc_std + fvc_mean

            # 3. Target: Unnormalize (dataset returns normalized targets)
            target_real = targets * fvc_std + fvc_mean

            all_targets.append(target_real.cpu().numpy())
            all_mus.append(mu_real.cpu().numpy())
            all_sigmas.append(sigma_real.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_mus = np.concatenate(all_mus)
    all_sigmas = np.concatenate(all_sigmas)

    # Calculate competition metric
    metric_score = calculate_metric(all_targets, all_mus, all_sigmas)

    return avg_loss, metric_score


def run_training():
    """
    Main training loop with differential learning rates and metric tracking.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()

    # 1. Load Data
    train_loader, val_loader, _, stats = get_dataloaders()

    # 2. Initialize Model
    device = Config.DEVICE
    model = RIDSNet().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the new heads
    backbone_params = list(model.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))

    # Head params are those not in backbone
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler & Loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX
    )
    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_metric = -float("inf")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")
    print(f"Backbone LR: {Config.LR_BACKBONE}, Head LR: {Config.LR_HEAD}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate_one_epoch(
            model, val_loader, criterion, device, stats
        )

        scheduler.step()

        epoch_time = time.time() - start_time

        # Save best model based on Metric (Higher is better)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Metric={val_metric} (New Best) [{epoch_time:.1f}s]"
            )
        else:
            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Metric={val_metric} [{epoch_time:.1f}s]"
            )

    print(f"Training complete. Best Validation Metric: {best_metric}")
