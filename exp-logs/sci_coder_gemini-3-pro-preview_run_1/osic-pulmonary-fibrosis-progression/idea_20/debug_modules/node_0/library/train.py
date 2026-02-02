import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.utils import get_device, get_logger, seed_everything, AverageMeter
from library.data import get_dataloaders
from library.model import ModalityAwareDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss

# Initialize Logger
logger = get_logger("train_module")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    loss_meter = AverageMeter("Loss")

    for batch in loader:
        # Move data to device
        img_axial = batch["img_axial"].to(device, dtype=torch.float32)
        img_coronal = batch["img_coronal"].to(device, dtype=torch.float32)
        tabular = batch["tabular"].to(device, dtype=torch.float32)
        meta = batch["meta"].to(device, dtype=torch.float32)
        target = batch["target"].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(img_axial, img_coronal, tabular, meta)

        # Calculate loss
        loss = criterion(pred_fvc, pred_sigma, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), img_axial.size(0))

    return loss_meter.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average metric (Negative Log Likelihood).
    """
    model.eval()
    loss_meter = AverageMeter("ValLoss")

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device, dtype=torch.float32)
            img_coronal = batch["img_coronal"].to(device, dtype=torch.float32)
            tabular = batch["tabular"].to(device, dtype=torch.float32)
            meta = batch["meta"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)

            pred_fvc, pred_sigma = model(img_axial, img_coronal, tabular, meta)

            loss = criterion(pred_fvc, pred_sigma, target)
            loss_meter.update(loss.item(), img_axial.size(0))

    # The metric is negative of the loss (higher is better)
    return -loss_meter.avg


def generate_submission(
    model, loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    logger.info("Generating submission...")

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device, dtype=torch.float32)
            img_coronal = batch["img_coronal"].to(device, dtype=torch.float32)
            tabular = batch["tabular"].to(device, dtype=torch.float32)
            meta = batch["meta"].to(device, dtype=torch.float32)
            patient_weeks = batch["patient_week"]

            pred_fvc, pred_sigma = model(img_axial, img_coronal, tabular, meta)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()

            for pw, fvc, sigma in zip(patient_weeks, pred_fvc, pred_sigma):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": sigma})

    df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")


def train_model(
    epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-2,
    patience=8,
    num_workers=4,
    seed=42,
):
    """
    Main training pipeline.
    """
    seed_everything(seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. Data Loading
    loaders = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, metadata_dir="./metadata"
    )

    # 2. Model Initialization
    model = ModalityAwareDualAxisNet(tabular_input_dim=7, embedding_dim=1280)
    model = model.to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = "./working/best_model.pth"

    logger.info("Starting training...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_metric = evaluate(model, loaders["val"], criterion, device)

        # Step scheduler
        scheduler.step()

        # Logging
        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved! Metric: {best_metric:.10f}")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # 5. Inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(
        model, loaders["test"], device, output_path="./submission/submission.csv"
    )
