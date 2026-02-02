import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data import get_dataloaders
from library.model import ASADAN, laplace_log_likelihood_loss
from library.utils import AverageMeter, score, seed_everything


class LaplaceLoss(nn.Module):
    """
    Wrapper for the functional Laplace Log Likelihood loss.
    """

    def __init__(self, max_error=1000, confidence_clip=70):
        super().__init__()
        self.max_error = max_error
        self.confidence_clip = confidence_clip

    def forward(self, y_true, y_pred, sigma):
        return laplace_log_likelihood_loss(
            y_true,
            y_pred,
            sigma,
            max_error=self.max_error,
            confidence_clip=self.confidence_clip,
        )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()

    losses = AverageMeter()
    scores = AverageMeter()

    for batch_idx, data in enumerate(loader):
        # Move inputs to device
        img_axial = data["img_axial"].to(device)
        img_coronal = data["img_coronal"].to(device)
        static_features = data["static_features"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)
        week = data["week"].to(device)
        target = data["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(
            img_axial, img_coronal, static_features, baseline_fvc, week
        )

        # Calculate loss
        loss = criterion(target, fvc_pred, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metric for monitoring (on CPU)
        metric_score = score(
            target.detach().cpu().numpy(),
            fvc_pred.detach().cpu().numpy(),
            sigma_pred.detach().cpu().numpy(),
            max_error=Config.MAX_ERROR,
            confidence_clip=Config.CONFIDENCE_CLIP,
        )

        # Update trackers
        losses.update(loss.item(), img_axial.size(0))
        scores.update(metric_score, img_axial.size(0))

    return losses.avg, scores.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # Move inputs to device
            img_axial = data["img_axial"].to(device)
            img_coronal = data["img_coronal"].to(device)
            static_features = data["static_features"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            week = data["week"].to(device)
            target = data["target"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                img_axial, img_coronal, static_features, baseline_fvc, week
            )

            # Calculate loss
            loss = criterion(target, fvc_pred, sigma_pred)

            # Calculate metric
            metric_score = score(
                target.cpu().numpy(),
                fvc_pred.cpu().numpy(),
                sigma_pred.cpu().numpy(),
                max_error=Config.MAX_ERROR,
                confidence_clip=Config.CONFIDENCE_CLIP,
            )

            # Update trackers
            losses.update(loss.item(), img_axial.size(0))
            scores.update(metric_score, img_axial.size(0))

    return losses.avg, scores.avg


def run_training(debug=False):
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    print(f"Debug mode: {debug}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=debug,
    )

    # 3. Model
    model = ASADAN()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss(
        max_error=Config.MAX_ERROR, confidence_clip=Config.CONFIDENCE_CLIP
    )

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print metrics (Full Precision)
        print(f"Epoch {epoch}/{Config.EPOCHS} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss} | Train Metric: {train_score}")
        print(f"  Val Loss:   {val_loss} | Val Metric:   {val_score}")

        # Early Stopping & Checkpointing
        # Metric is negative Log Likelihood, higher is better (closer to 0)
        if val_score > best_score:
            print(
                f"  [Improved] Score improved from {best_score} to {val_score}. Saving model..."
            )
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation Score: {best_score}")
    print(f"Best model saved to: {best_model_path}")

    return best_model_path
