import os
import torch
import numpy as np
import random
import time
from library.config import Config
from library.dataset import get_dataloaders
from library.model import H2DAN
from library.loss import LaplaceLogLikelihoodLoss


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move batch data to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model returns dict with 'fvc_pred', 'confidence_pred', etc.
        outputs = model(batch)

        pred_fvc = outputs["fvc_pred"]
        pred_sigma = outputs["confidence_pred"]
        target_fvc = batch["target"]

        # Compute loss
        loss = criterion(pred_fvc, pred_sigma, target_fvc)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average metric score (Metric = -Loss).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Move batch data to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = model(batch)

            pred_fvc = outputs["fvc_pred"]
            pred_sigma = outputs["confidence_pred"]
            target_fvc = batch["target"]

            # Compute loss
            loss = criterion(pred_fvc, pred_sigma, target_fvc)

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # The competition metric is negative of the Laplace Log Likelihood Loss
    # Metric = -Loss
    avg_metric = -avg_loss
    return avg_metric


def run_training():
    """
    Main execution function for training the H2-DAN model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Backbone: {Config.BACKBONE}")

    # 2. Data Loading
    train_loader, val_loader, scaler = get_dataloaders()

    # 3. Model Initialization
    # We need to determine the input dimension for the tabular encoder.
    # The scaler transforms the deep features. We can check the shape from the scaler.
    # num_features = len(num_cols) + sum(len(cats) for cats in onehot.categories_)
    # Alternatively, we can inspect a sample from the dataset.
    sample_batch = next(iter(train_loader))
    tabular_dim = sample_batch["deep_tab"].shape[1]

    model = H2DAN(tabular_input_dim=tabular_dim)
    model.to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss()
    criterion.to(device)

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metric = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Metric: {val_metric:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Metric: {best_metric:.8f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Metric: {best_metric:.8f}")
    return best_model_path
