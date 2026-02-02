import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, AverageMeter, calculate_metric
from library.loss import MetricAlignedLaplaceLoss
from library.data import get_dataloaders
from library.model import CRDSNet


def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Returns [mu_scaled, sigma_scaled]
        preds = model(images, tabular)

        # Calculate loss in standardized space
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes both the standardized loss and the original scale metric.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    # Constants for inverse transformation
    target_mean = Config.TARGET_MEAN
    target_std = Config.TARGET_STD

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images, tabular)

            # 1. Calculate Loss (Standardized Space)
            loss = criterion(preds, targets)
            loss_meter.update(loss.item(), images.size(0))

            # 2. Calculate Metric (Original Space)
            # Inverse transform predictions
            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()

            mu_pred = mu_scaled * target_std + target_mean
            sigma_pred = sigma_scaled * target_std

            # Inverse transform targets
            targets_np = targets.cpu().numpy()
            targets_orig = targets_np * target_std + target_mean

            # Calculate metric
            score = calculate_metric(targets_orig, mu_pred, sigma_pred)
            metric_meter.update(score, images.size(0))

    return loss_meter.avg, metric_meter.avg


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    target_mean = Config.TARGET_MEAN
    target_std = Config.TARGET_STD

    with torch.no_grad():
        for images, tabular, pat_week_ids in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            preds = model(images, tabular)

            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()

            # Inverse transform
            mu_pred = mu_scaled * target_std + target_mean
            sigma_pred = sigma_scaled * target_std

            # Post-processing: Hard clip sigma at 70ml
            sigma_pred = np.maximum(sigma_pred, 70)

            for i, pat_week in enumerate(pat_week_ids):
                results.append(
                    {
                        "Patient_Week": pat_week,
                        "FVC": mu_pred[i],
                        "Confidence": sigma_pred[i],
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function for training, validation, and submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 2. Model
    model = CRDSNet().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Filter parameters to ensure we only optimize those requiring grad
    backbone_ids = list(map(id, model.backbone.parameters()))
    head_params = filter(
        lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters()
    )
    backbone_params = filter(lambda p: p.requires_grad, model.backbone.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler & Loss
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)
    criterion = MetricAlignedLaplaceLoss()

    # 5. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best metric! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)


if __name__ == "__main__":
    run_training()
