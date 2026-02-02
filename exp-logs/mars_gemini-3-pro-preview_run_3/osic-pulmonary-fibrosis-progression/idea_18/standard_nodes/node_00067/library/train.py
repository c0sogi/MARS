import os
import torch
import numpy as np
import torch.nn.functional as F
from library.config import Config
from library import utils, data, model


def run_training(debug=False):
    """
    Executes the training pipeline for the CLR-Net model.

    Args:
        debug (bool): If True, uses a subset of data for quick debugging.

    Returns:
        float: The best validation metric score achieved.
    """
    # 1. Setup
    Config.setup()
    utils.seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, _, _ = data.get_dataloaders(debug=debug)

    # 3. Model Initialization
    print("Initializing model...")
    net = model.CLRNet().to(device)

    # 4. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest (heads, streams)
    backbone_ids = list(map(id, net.backbone.parameters()))
    rest_params = filter(lambda p: id(p) not in backbone_ids, net.parameters())

    # Only optimize backbone parameters that require grad (some are frozen)
    backbone_params = filter(lambda p: p.requires_grad, net.backbone.parameters())

    param_groups = [
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": rest_params, "lr": Config.LR_HEAD},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    criterion = utils.LaplaceLogLikelihoodLoss().to(device)

    # 5. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        net.train()
        train_losses = []

        for batch_idx, (imgs, clin_data, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            clin_data = clin_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = net(imgs, clin_data)

            # Loss calculation (in standardized space)
            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        scheduler.step()

        # --- Validation Phase ---
        net.eval()
        val_preds_mu = []
        val_preds_sigma = []
        val_targets = []

        with torch.no_grad():
            for imgs, clin_data, targets in val_loader:
                imgs = imgs.to(device)
                clin_data = clin_data.to(device)
                targets = targets.to(device)

                # Forward pass
                # preds: [mean, raw_sigma]
                preds = net(imgs, clin_data)

                mu_std = preds[:, 0]
                raw_sigma = preds[:, 1]

                # Apply softplus to get positive sigma in standardized space
                sigma_std = F.softplus(raw_sigma) + 1e-6

                # Store standardized values
                val_preds_mu.extend(mu_std.cpu().numpy())
                val_preds_sigma.extend(sigma_std.cpu().numpy())
                val_targets.extend(targets.cpu().numpy().flatten())

        # --- Metric Calculation (in original ml space) ---
        val_preds_mu = np.array(val_preds_mu)
        val_preds_sigma = np.array(val_preds_sigma)
        val_targets = np.array(val_targets)

        # Inverse transform predictions
        pred_mu_ml, pred_sigma_ml = utils.inverse_transform(
            val_preds_mu, val_preds_sigma
        )

        # Inverse transform targets (they were standardized in dataset)
        # target_ml = target_std * STD + MEAN
        target_ml = val_targets * Config.TARGET_STD + Config.TARGET_MEAN

        # Calculate competition metric
        # Note: calculate_metric handles the clipping internally
        current_metric = utils.calculate_metric(target_ml, pred_mu_ml, pred_sigma_ml)

        # --- Logging & Checkpointing ---
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Metric: {current_metric}"
        )

        if current_metric > best_metric:
            print(
                f"New best metric! ({best_metric} -> {current_metric}). Saving model..."
            )
            best_metric = current_metric
            torch.save(net.state_dict(), best_model_path)

    print(f"Training complete. Best Validation Metric: {best_metric}")
    return best_metric
