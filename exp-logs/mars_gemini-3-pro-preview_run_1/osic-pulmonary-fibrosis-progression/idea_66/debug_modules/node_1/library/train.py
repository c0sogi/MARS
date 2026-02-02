import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, AverageMeter, metric_function
from library.data import get_dataloaders
from library.model import (
    TSCPNet,
    train_one_epoch,
    validate,
    laplace_log_likelihood_loss,
)


class LaplaceLikelihoodLoss(nn.Module):
    """
    PyTorch Module wrapper for the modified Laplace Log Likelihood loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, params, baseline_fvc, week_diff, true_fvc):
        return laplace_log_likelihood_loss(params, baseline_fvc, week_diff, true_fvc)


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    device_name=Config.DEVICE,
    checkpoint_dir=Config.CHECKPOINT_DIR,
    debug=False,
):
    """
    Orchestrates the training process with hyperparameter flexibility.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for dataloaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.
        device_name (str): Device to run training on ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
        debug (bool): If True, could be used to limit dataset size (relies on dataloader config).

    Returns:
        float: The best validation score achieved.
    """
    set_seed(Config.SEED)
    device = torch.device(device_name)

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get Dataloaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Initialize Model
    model = TSCPNet().to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Training Loop Variables
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Run Training Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Run Validation Step
        val_score = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Checkpointing and Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_score


def generate_predictions(
    model_path=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device_name=Config.DEVICE,
):
    """
    Generates predictions for the test set using the trained model and saves to CSV.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device_name (str): Device to run inference on.
    """
    device = torch.device(device_name)

    # Load Test Data
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize and Load Model
    model = TSCPNet().to(device)
    if not os.path.exists(model_path):
        print(
            f"Error: Model file not found at {model_path}. Cannot generate predictions."
        )
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            meta = batch["meta"].to(device)
            week_diff = batch["week_diff"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]
            weeks = batch["week"]

            # Forward pass
            params = model(img_ax, img_cor, meta)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Calculate Trajectories
            pred_fvc = baseline_fvc + alpha * week_diff
            confidence = sigma_base + sigma_growth * torch.abs(week_diff)

            # Move results to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            confidence = confidence.cpu().numpy()
            weeks = weeks.cpu().numpy()

            # Aggregate results
            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(weeks[i])
                fvc = pred_fvc[i]
                conf = confidence[i]

                patient_week = f"{pid}_{wk}"
                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
