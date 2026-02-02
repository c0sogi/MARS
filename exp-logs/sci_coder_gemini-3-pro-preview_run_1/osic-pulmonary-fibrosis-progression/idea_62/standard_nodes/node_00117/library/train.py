import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import NDSSLN


def criterion(fvc_true, fvc_pred, sigma, device):
    """
    Differentiable implementation of the modified Laplace Log Likelihood loss.
    Loss = -Metric, since we want to maximize the metric.
    """
    # Constants
    sq2 = torch.sqrt(torch.tensor(2.0, device=device))

    # Clip Sigma (Confidence) at 70ml
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE_CLIP)

    # Calculate Absolute Error and Clip at 1000ml
    delta = torch.abs(fvc_true - fvc_pred)
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR_CLIP)

    # Metric Formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    # Loss (to minimize) = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
    loss = (sq2 * delta_clipped) / sigma_clipped + torch.log(sq2 * sigma_clipped)

    return torch.mean(loss)


def train_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)

        base_fvc = batch["base_fvc"].to(device)
        delta_week = batch["delta_week"].to(device)
        target_fvc = batch["target"].to(device)

        batch_size = img_ax.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        # Model returns parameters for the trajectory
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Calculate Predictions based on trajectory
        # FVC = Baseline + alpha * (Current_Week - Baseline_Week)
        fvc_pred = base_fvc + alpha * delta_week

        # Confidence = Sigma_base + Sigma_growth * |Current_Week - Baseline_Week|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        # Calculate Loss
        loss = criterion(target_fvc, fvc_pred, sigma_pred, device)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    return running_loss / dataset_size


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()

    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            target_fvc = batch["target"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate Predictions
            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Collect results for full metric calculation
            all_true.append(target_fvc.cpu().numpy())
            all_pred.append(fvc_pred.cpu().numpy())
            all_sigma.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    sigma = np.concatenate(all_sigma)

    # Calculate official metric
    score = laplace_log_likelihood(y_true, y_pred, sigma)
    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            patient_weeks = batch["patient_week"]  # List of ID strings

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate Predictions
            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU and flatten
            fvc_pred = fvc_pred.cpu().numpy().flatten()
            sigma_pred = sigma_pred.cpu().numpy().flatten()

            # Store results
            for pw, f, s in zip(patient_weeks, fvc_pred, sigma_pred):
                results.append({"Patient_Week": pw, "FVC": f, "Confidence": s})

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Main execution function for training, validation, and submission.
    """
    # Set reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Initialize Model
    model = NDSSLN().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop Variables
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Score: {val_score}"
        )

        # Checkpoint and Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # Load best model for submission
    if os.path.exists(best_model_path):
        print("Loading best model for submission generation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
