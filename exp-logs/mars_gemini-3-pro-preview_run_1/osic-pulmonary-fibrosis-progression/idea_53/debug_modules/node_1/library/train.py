import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    LaplaceLogLikelihoodLoss,
    calculate_metric,
    EarlyStopping,
)
from library.data import get_dataloaders
from library.model import NSLHN


def train_epoch(loader, model, criterion, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        rel_week = batch["relative_week"].to(device)
        base_fvc = batch["baseline_fvc"].to(device)
        target = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

        # Calculate loss
        loss = criterion(pred_fvc, pred_sigma, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def validate(loader, model, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["relative_week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

            # Calculate metric
            # Note: We compute metric per batch and average, or compute on full set.
            # Given the metric is an average over patients, averaging batch means is acceptable approximation
            # provided batch sizes are roughly equal, but exact calculation is safer.
            # Here we calculate mean metric per batch for tracking.
            metric = calculate_metric(pred_fvc.cpu(), pred_sigma.cpu(), target.cpu())
            scores.update(metric, img_ax.size(0))

    return scores.avg


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")
    model.eval()

    predictions = []

    # We need to align predictions with Patient_Week IDs.
    # The test_loader is sequential (shuffle=False) and based on Config.TEST_CSV.
    # We will read the CSV to get the IDs.
    test_df = pd.read_csv(Config.TEST_CSV)
    patient_weeks = test_df["Patient_Week"].values

    pred_fvcs = []
    pred_sigmas = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["relative_week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

            pred_fvcs.extend(pred_fvc.cpu().numpy())
            pred_sigmas.extend(pred_sigma.cpu().numpy())

    # Clip confidence values at 70ml as per metric requirements
    pred_sigmas = np.array(pred_sigmas)
    pred_sigmas = np.maximum(pred_sigmas, 70.0)

    # Create DataFrame
    sub_df = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": pred_fvcs, "Confidence": pred_sigmas}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(debug=False, epochs=None):
    """
    Main driver for training, validation, and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(f"Starting training for experiment: {Config.EXP_ID}")
    print(f"Device: {device}")

    # 2. Data
    # If debug is enabled, we could theoretically limit batches, but here we run full logic
    # relying on the speed of the environment.
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model
    model = NSLHN()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss(
        clip_sigma=Config.CONFIDENCE_CLIP,
        clip_error=Config.MAX_ERROR,
        apply_error_clip_in_train=False,  # Let model see gradients from large errors
    )

    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="max", delta=0.001)

    # 5. Training Loop
    num_epochs = epochs if epochs is not None else Config.EPOCHS

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_score = validate(val_loader, model, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Score: {val_score}"
        )

        # Early Stopping check
        early_stopping(val_score, model, Config.MODEL_SAVE_PATH)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device)
    print("Training and inference complete.")


if __name__ == "__main__":
    # This block is excluded as per instructions, but the function run_training is available.
    pass
