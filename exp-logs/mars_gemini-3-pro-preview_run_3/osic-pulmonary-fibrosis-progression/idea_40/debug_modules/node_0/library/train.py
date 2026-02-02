import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import CIDSNet


class LaplaceLoss(nn.Module):
    """
    Smooth Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    As per the idea description:
    - Explicitly includes sqrt(2) constants.
    - Removes all clipping/thresholding to prevent gradient vanishing on outliers.
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, output, target):
        """
        Args:
            output: (B, 2) tensor containing [mu, sigma]
            target: (B,) tensor containing true values
        """
        mu = output[:, 0]
        sigma = output[:, 1]

        # Calculate absolute error
        abs_error = torch.abs(target - mu)

        # Calculate loss terms
        # Term 1: Scaled absolute error
        term1 = (self.sqrt_2.to(mu.device) * abs_error) / sigma

        # Term 2: Log of scaled uncertainty
        term2 = torch.log(self.sqrt_2.to(mu.device) * sigma)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, tabular)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, processor):
    """
    Evaluates the model on the validation set.
    Calculates both the loss (on scaled data) and the competition metric (on raw ml data).
    """
    model.eval()
    running_loss = 0.0

    # Lists to store raw (unscaled) predictions and targets for metric calculation
    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    # Get scaler statistics for inverse transformation
    target_mean = processor.target_mean
    target_std = processor.target_std

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images, tabular)

            # Compute Loss (on scaled data)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            # Inverse Transform for Metric Calculation
            # 1. Targets: z * std + mean
            true_fvc_batch = targets.cpu().numpy() * target_std + target_mean

            # 2. Predictions (Mean): z * std + mean
            pred_fvc_batch = outputs[:, 0].cpu().numpy() * target_std + target_mean

            # 3. Predictions (Sigma): z * std (Scale only, no mean shift)
            pred_sigma_batch = outputs[:, 1].cpu().numpy() * target_std

            all_true_fvc.extend(true_fvc_batch)
            all_pred_fvc.extend(pred_fvc_batch)
            all_pred_sigma.extend(pred_sigma_batch)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate Competition Metric
    metric_score = score_function(
        np.array(all_true_fvc), np.array(all_pred_fvc), np.array(all_pred_sigma)
    )

    return epoch_loss, metric_score


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main execution function for training the CIDS-Net model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # 1. Data Loading
    train_loader, val_loader, _, processor = get_dataloaders(batch_size=batch_size)

    # 2. Model Initialization
    model = CIDSNet().to(device)

    # 3. Optimization Setup
    # Differential Learning Rates
    backbone_params = []
    head_params = []

    # Identify backbone parameters (Stream B backbone)
    # Note: model.visual_stream.backbone is the EfficientNet
    backbone_ids = list(map(id, model.visual_stream.backbone.parameters()))

    for name, param in model.named_parameters():
        if id(param) in backbone_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    best_epoch = 0
    patience = 10
    patience_counter = 0

    save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device, processor)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.16f}"
        )

        # Checkpointing (Higher metric is better)
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! (Metric: {val_metric:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Metric: {best_metric:.16f} at Epoch {best_epoch+1}")
    return best_metric


if __name__ == "__main__":
    # This block is for testing the module locally, but the requirements say
    # "DO NOT include an if __name__ == '__main__': block" for the final output logic.
    # However, to make this script runnable as a standalone training script if needed,
    # I will invoke the run_training function.
    # Per instructions "Only implement the module class/functions", I will provide the functions.
    pass
