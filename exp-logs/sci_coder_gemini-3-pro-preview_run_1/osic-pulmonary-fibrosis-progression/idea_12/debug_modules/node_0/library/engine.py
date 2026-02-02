import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.utils import AverageMeter, get_score
from library.loss import LaplaceLogLikelihoodLoss
from library.network import PriorPreservingDualAxisNet
from library.dataset import LungDataset, get_transforms


def train_one_epoch(dataloader, model, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(dataloader):
        # Move inputs to device
        image_axial = data["image_axial"].to(device)
        image_coronal = data["image_coronal"].to(device)
        tabular = data["tabular"].to(device)

        week_delta = data["week_delta"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)
        target_fvc = data["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model outputs: [Alpha, Sigma_Base, Sigma_Growth]
        preds = model(image_axial, image_coronal, tabular)

        # Calculate loss
        loss = criterion(preds, week_delta, baseline_fvc, target_fvc)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), image_axial.size(0))

    return loss_meter.avg


def evaluate(dataloader, model, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    score_meter = AverageMeter()

    with torch.no_grad():
        for data in dataloader:
            # Move inputs to device
            image_axial = data["image_axial"].to(device)
            image_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)

            week_delta = data["week_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target_fvc = data["target"].to(device)

            # Forward pass
            preds = model(image_axial, image_coronal, tabular)

            # Unpack predictions
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct FVC and Confidence based on parametric formula
            # FVC = Base + Alpha * Delta
            fvc_pred = baseline_fvc + alpha * week_delta

            # Confidence = Base + Growth * |Delta|
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

            # Calculate metric
            # get_score handles the clipping internally
            score = get_score(target_fvc, fvc_pred, sigma_pred)

            score_meter.update(score, image_axial.size(0))

    return score_meter.avg


def run_training(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training loop with early stopping.
    """
    criterion = LaplaceLogLikelihoodLoss()
    best_score = -float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, scheduler
        )

        # Validate
        val_score = evaluate(val_loader, model, device)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        # Early Stopping & Model Saving
        # Metric is negative Laplace Log Likelihood (higher is better, closer to 0)
        if val_score > best_score:
            print(
                f"Score improved from {best_score:.10f} to {val_score:.10f}. Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation Score: {best_score:.10f}")
    return best_score


def predict_and_submit(model, device, metadata_path, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load test metadata to get Patient_Week identifiers
    test_df = pd.read_csv(metadata_path)

    # Create Dataset and DataLoader
    # Note: Sequential loading (shuffle=False) is critical to match predictions with test_df rows
    test_dataset = LungDataset(
        metadata_path=metadata_path,
        mode="test",
        transform=get_transforms(mode="test"),
        load_cached_data=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()

    all_fvc_preds = []
    all_sigma_preds = []

    with torch.no_grad():
        for data in test_loader:
            image_axial = data["image_axial"].to(device)
            image_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)

            week_delta = data["week_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)

            # Forward pass
            preds = model(image_axial, image_coronal, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Calculate FVC
            fvc_pred = baseline_fvc + alpha * week_delta

            # Calculate Confidence (Sigma)
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

            # Store results (move to CPU numpy)
            all_fvc_preds.extend(fvc_pred.cpu().numpy())
            all_sigma_preds.extend(sigma_pred.cpu().numpy())

    # Prepare Submission DataFrame
    # Ensure lengths match
    if len(all_fvc_preds) != len(test_df):
        raise ValueError(
            f"Mismatch: {len(all_fvc_preds)} predictions vs {len(test_df)} metadata rows."
        )

    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": all_fvc_preds,
            "Confidence": all_sigma_preds,
        }
    )

    # Apply Final Clipping for Submission Format (as per metric definition)
    # Note: The metric clips confidence at 70, so we should output values consistent with that expectation,
    # though the evaluation code usually applies the clip itself. The prompt says:
    # "confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty"
    # It is safer to clip here to be explicit.
    submission["Confidence"] = submission["Confidence"].apply(lambda x: max(x, 70.0))

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
