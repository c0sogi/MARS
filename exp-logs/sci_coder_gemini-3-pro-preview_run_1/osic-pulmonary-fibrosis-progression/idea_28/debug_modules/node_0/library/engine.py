import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.utils import AverageMeter, LaplaceLogLikelihood, seed_everything
from library.dataset import FVCDataset, get_transforms
from library.model import VERNet


def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch and move to device
        img_ax = batch["img_axial"].to(device)
        img_cor = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        week = batch["week"].to(device)
        base_fvc = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

        # Calculate loss
        loss = criterion(pred_fvc, pred_sigma, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    if scheduler:
        scheduler.step()

    return losses.avg


def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns the average metric (Negative Laplace Log Likelihood).
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

            # The criterion calculates the Loss. The metric is -Loss.
            loss = criterion(pred_fvc, pred_sigma, target)

            # We want to maximize the metric, so we track -loss
            metric_meter.update(-loss.item(), img_ax.size(0))

    return metric_meter.avg


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a DataFrame with columns: Patient_Week, FVC, Confidence.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

            # Post-processing: Clip confidence at 70ml as per metric definition
            # Although the metric function clips it, the submission should also be reasonable.
            pred_sigma = torch.clamp(pred_sigma, min=70.0)

            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, pred_fvc_np, pred_sigma_np):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    return pd.DataFrame(results)


def run_training(
    epochs=30,
    batch_size=16,
    learning_rate=3e-4,
    device="cuda",
    save_path="./working/best_model.pth",
    patience=8,
    num_workers=2,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    seed_everything(42)

    # Ensure working directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize Datasets and Loaders
    train_dataset = FVCDataset(mode="train", transform=get_transforms("train"))
    val_dataset = FVCDataset(mode="val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = VERNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Loss Function
    criterion = LaplaceLogLikelihood()

    # Training Loop Variables
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion, scheduler
        )
        val_metric = evaluate(model, val_loader, device, criterion)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Metric: {val_metric}"
        )

        # Early Stopping and Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New Best Model Saved! Metric: {best_metric}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Validation Metric: {best_metric}")
    return best_metric


def generate_submission_file(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    device="cuda",
    batch_size=16,
    num_workers=2,
):
    """
    Generates the submission file using the trained model.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Test Data
    test_dataset = FVCDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Model
    model = VERNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(
            f"WARNING: Model path {model_path} does not exist. Using random initialization."
        )

    # Generate Predictions
    df_submission = predict(model, test_loader, device)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")
