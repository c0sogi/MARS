import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import AverageMeter, LaplaceLogLikelihood, seed_everything
from library.data import get_dataloaders
from library.model import DualAxisTransformer


def train_one_epoch(epoch, model, loader, optimizer, device):
    """
    Handles the training of a single epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)  # [delta_week, base_fvc]
        target = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Output: [alpha, sigma_base, sigma_growth]
        preds = model(img_ax, img_cor, tabular)

        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # Extract metadata for parametric equation
        delta_week = meta[:, 0]
        base_fvc = meta[:, 1]

        # Calculate predicted FVC and Confidence
        # FVC = Base + alpha * delta_t
        fvc_pred = base_fvc + alpha * delta_week

        # Confidence = sigma_base + sigma_growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        # Compute Loss
        # The metric function handles the clipping logic internally
        loss = LaplaceLogLikelihood(target, fvc_pred, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    score_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            preds = model(img_ax, img_cor, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            delta_week = meta[:, 0]
            base_fvc = meta[:, 1]

            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Metric calculation (Negative LLL, higher is better, but function returns negative value)
            # The function returns the negative term (loss), so we want to minimize this loss.
            # Wait, the util function returns: -term1 - term2.
            # This is the Metric value (negative). Higher (closer to 0) is better.
            score = LaplaceLogLikelihood(target, fvc_pred, sigma_pred)

            score_meter.update(score.item(), img_ax.size(0))

    return score_meter.avg


def run_training():
    """
    Main orchestration function for training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(debug=Config.DEBUG)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    model = DualAxisTransformer()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training Loop
    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score}"
        )

        # Early Stopping & Checkpointing
        # Note: The metric is negative, so higher is better (e.g., -6.5 is better than -6.8)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Score: {best_score} at Epoch {best_epoch}")


def inference():
    """
    Loads the best model and generates predictions for the test set.
    """
    print("Starting inference...")
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Load Model
    model = DualAxisTransformer()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            patient_weeks = batch["patient_week"]

            preds = model(img_ax, img_cor, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            delta_week = meta[:, 0]
            base_fvc = meta[:, 1]

            # Calculate predictions
            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for i, pw in enumerate(patient_weeks):
                # Apply clipping to confidence for submission consistency
                # Although metric handles it, submission should be reasonable
                conf = max(sigma_pred[i], 70.0)

                results.append(
                    {"Patient_Week": pw, "FVC": fvc_pred[i], "Confidence": conf}
                )

    # Create DataFrame
    submission = pd.DataFrame(results)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Run Pipeline
    run_training()
    inference()
