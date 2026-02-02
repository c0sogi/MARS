import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import VisuallyContextualizedNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the Modified Laplace Log Likelihood Loss.
    Explicitly retains error clipping (max 1000ml) and confidence clipping (min 70ml)
    during optimization to align the loss landscape with the competition metric.
    """

    def __init__(
        self, max_error=Config.MAX_ERROR, min_confidence=Config.MIN_CONFIDENCE
    ):
        super().__init__()
        self.max_error = max_error
        self.min_confidence = min_confidence

    def forward(self, fvc_pred, confidence_pred, target):
        # Clip confidence values to avoid division by zero or extremely small sigmas
        # Note: Model output is already Softplus, so it is positive, but we enforce min 70.
        sigma_clipped = torch.clamp(confidence_pred, min=self.min_confidence)

        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)

        # Clip error at 1000ml to reduce outlier impact
        delta = torch.clamp(abs_error, max=self.max_error)

        # Calculate Metric: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        sqrt_2 = 1.41421356
        metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

        # We want to maximize the metric, so we minimize the negative mean metric
        loss = -torch.mean(metric)

        return loss


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        weeks = batch["weeks"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        output = model(img_axial, img_coronal, tabular, weeks)
        fvc_pred = output["fvc"]
        confidence_pred = output["confidence"]

        loss = criterion(fvc_pred, confidence_pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img_axial.size(0)

    if scheduler:
        scheduler.step()

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            target = batch["target"].to(device)

            output = model(img_axial, img_coronal, tabular, weeks)

            all_targets.append(target.cpu().numpy())
            all_preds.append(output["fvc"].cpu().numpy())
            all_sigmas.append(output["confidence"].cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_sigmas = np.concatenate(all_sigmas)

    # Use the utility score function for consistent evaluation
    score = score_function(all_targets, all_preds, all_sigmas)
    return score


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            patient_weeks = batch["patient_week"]

            output = model(img_axial, img_coronal, tabular, weeks)

            fvc_preds = output["fvc"].cpu().numpy()
            conf_preds = output["confidence"].cpu().numpy()

            # Ensure confidence is clipped for submission as per rules
            conf_preds = np.maximum(conf_preds, Config.MIN_CONFIDENCE)

            for pw, fvc, conf in zip(patient_weeks, fvc_preds, conf_preds):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    df_sub = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(load_cached_data=True, epochs=Config.EPOCHS):
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Load Data
    # Note: get_dataloaders handles caching logic internally via process_patient
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    model = VisuallyContextualizedNet().to(device)

    # Loss, Optimizer, Scheduler
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Training Loop Variables
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_score = validate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print("New Best Model Saved!")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation Score: {best_score}")

    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device)


def main(load_cached_data=True):
    train_model(load_cached_data=load_cached_data)
