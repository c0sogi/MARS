import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, LaplaceLogLikelihoodLoss, seed_everything
from library.data import get_dataloaders
from library.model import BBSLNet


def train_fn(dataloader, model, optimizer, device, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in dataloader:
        # Unpack batch and move to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        time_delta = batch["time_delta"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, 2) -> [FVC_pred, Sigma_pred]
        preds = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

        # Compute loss
        loss = loss_fn(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, device, loss_fn):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            targets = batch["target"].to(device)

            preds = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)
            loss = loss_fn(preds, targets)

            loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def inference_fn(dataloader, model, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            # Forward pass
            # preds: (Batch, 2) -> [FVC, Confidence]
            preds = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

            fvc_preds = preds[:, 0].cpu().numpy()
            conf_preds = preds[:, 1].cpu().numpy()

            for i, pw in enumerate(patient_weeks):
                # Clip confidence at 70 ml as per metric requirements
                # The metric uses max(sigma, 70), so we ensure our submission reflects this.
                conf = max(conf_preds[i], 70.0)

                results.append(
                    {"Patient_Week": pw, "FVC": fvc_preds[i], "Confidence": conf}
                )

    # Create submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def run():
    """
    Main execution pipeline: Setup, Training with Early Stopping, and Inference.
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # Load Data
    train_loader, val_loader, sub_loader = get_dataloaders()

    # Initialize Model
    model = BBSLNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.scheduler_T_max
    )

    # Loss Function
    loss_fn = LaplaceLogLikelihoodLoss()

    # Early Stopping Variables
    best_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, loss_fn)

        # Validation
        val_loss = eval_fn(val_loader, model, device, loss_fn)

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.model_save_path)
            print("  -> Validation improved. Model saved.")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_loss}")

    # 2. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    print("Generating submission...")
    inference_fn(sub_loader, model, device)
