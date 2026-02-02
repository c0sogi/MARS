import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, LaplaceLogLikelihoodLoss, calculate_metric
from library.model import NBCSLN
from library.data import get_dataloaders


def train_fn(dataloader, model, optimizer, device, loss_fn):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in dataloader:
        # Move inputs to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device).squeeze(-1)  # (B,)

        optimizer.zero_grad()

        # Forward pass: Get trajectory parameters
        # outputs: [alpha, sigma_base, sigma_growth]
        outputs = model(img_axial, img_coronal, tabular)

        alpha = outputs[:, 0]
        sigma_base = outputs[:, 1]
        sigma_growth = outputs[:, 2]

        # Reconstruct predictions based on metadata
        # meta: [baseline_fvc, week_diff]
        baseline_fvc = meta[:, 0]
        week_diff = meta[:, 1]

        # FVC_pred = Base + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * week_diff

        # Sigma_pred = Base + growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

        # Calculate Loss
        loss = loss_fn(fvc_pred, sigma_pred, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_axial.size(0))

    return losses.avg


def eval_fn(dataloader, model, device):
    """
    Runs validation and calculates the competition metric.
    """
    model.eval()
    metrics = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device).squeeze(-1)

            # Forward pass
            outputs = model(img_axial, img_coronal, tabular)

            alpha = outputs[:, 0]
            sigma_base = outputs[:, 1]
            sigma_growth = outputs[:, 2]

            # Reconstruct
            baseline_fvc = meta[:, 0]
            week_diff = meta[:, 1]

            fvc_pred = baseline_fvc + alpha * week_diff
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

            # Calculate Metric
            score = calculate_metric(fvc_pred, sigma_pred, target)
            metrics.update(score, img_axial.size(0))

    return metrics.avg


def inference_fn(dataloader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    results = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in dataloader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            patient_weeks = batch["patient_week"]  # List of strings

            # Forward pass
            outputs = model(img_axial, img_coronal, tabular)

            alpha = outputs[:, 0]
            sigma_base = outputs[:, 1]
            sigma_growth = outputs[:, 2]

            # Reconstruct
            baseline_fvc = meta[:, 0]
            week_diff = meta[:, 1]

            fvc_pred = baseline_fvc + alpha * week_diff
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

            # Clip confidence for submission consistency
            sigma_pred = torch.clamp(sigma_pred, min=Config.CONFIDENCE_MIN)

            # Store results
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for i, pw in enumerate(patient_weeks):
                results.append(
                    {
                        "Patient_Week": pw,
                        "FVC": fvc_pred[i],
                        "Confidence": sigma_pred[i],
                    }
                )

    return pd.DataFrame(results)


def run():
    """
    Main execution function.
    """
    # 1. Setup
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model
    model = NBCSLN()
    model.to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    loss_fn = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, loss_fn)

        # Validate
        val_metric = eval_fn(val_loader, model, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best metric! Model saved.")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    sub_df = inference_fn(test_loader, model, device)

    # 7. Save Submission
    # Ensure columns are in correct order
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(sub_df.head())
