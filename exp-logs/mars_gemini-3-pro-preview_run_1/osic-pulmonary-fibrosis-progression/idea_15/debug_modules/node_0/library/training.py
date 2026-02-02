import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import AverageMeter, calculate_metric, seed_everything
from library.loss import RobustLaplaceLogLikelihoodLoss
from library.model import GranularTabularNetwork
from library.dataset import LungDataset


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    """
    Handles the training of one epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    for batch in loader:
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)
        smoke = batch["smoke"].to(device)
        percent = batch["percent"].to(device)
        priors = batch["priors"].to(device)
        time_delta = batch["time_delta"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, conf_pred = model(
            axial, coronal, age, sex, smoke, percent, priors, time_delta
        )

        # Loss calculation
        loss = loss_fn(fvc_pred, conf_pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = axial.size(0)
        loss_meter.update(loss.item(), batch_size)

        # Calculate metric score
        score = calculate_metric(target, fvc_pred, conf_pred)
        metric_meter.update(score, batch_size)

    # Step scheduler if it is per-epoch
    if scheduler:
        scheduler.step()

    return loss_meter.avg, metric_meter.avg


def validate_one_epoch(model, loader, loss_fn, device):
    """
    Handles the validation of one epoch.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            percent = batch["percent"].to(device)
            priors = batch["priors"].to(device)
            time_delta = batch["time_delta"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, conf_pred = model(
                axial, coronal, age, sex, smoke, percent, priors, time_delta
            )

            # Loss calculation
            loss = loss_fn(fvc_pred, conf_pred, target)

            # Metrics
            batch_size = axial.size(0)
            loss_meter.update(loss.item(), batch_size)

            # Calculate metric score
            score = calculate_metric(target, fvc_pred, conf_pred)
            metric_meter.update(score, batch_size)

    return loss_meter.avg, metric_meter.avg


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Load Test Dataset
    test_dataset = LungDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            percent = batch["percent"].to(device)
            priors = batch["priors"].to(device)
            time_delta = batch["time_delta"].to(device)
            patient_weeks = batch["patient_week"]

            # Forward pass
            fvc_pred, conf_pred = model(
                axial, coronal, age, sex, smoke, percent, priors, time_delta
            )

            # Collect results
            fvc_pred = fvc_pred.cpu().numpy()
            conf_pred = conf_pred.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, fvc_pred, conf_pred):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to file
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")

    # Debugging: Subset data if needed
    if Config.DEBUG:
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLES} samples.")
        indices_train = torch.randperm(len(train_dataset))[: Config.DEBUG_SAMPLES]
        train_dataset = Subset(train_dataset, indices_train)

        indices_val = torch.randperm(len(val_dataset))[: Config.DEBUG_SAMPLES]
        val_dataset = Subset(val_dataset, indices_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = GranularTabularNetwork()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    loss_fn = RobustLaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.N_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_score = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scheduler
        )

        # Validate
        val_loss, val_score = validate_one_epoch(model, val_loader, loss_fn, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{Config.N_EPOCHS} | Time: {elapsed:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Score: {train_score:.4f}")
        # Print full precision as requested
        print(f"Val Loss: {val_loss:.4f} | Val Score: {val_score}")

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Score! Model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # 6. Inference
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Generate submission
    generate_submission(model, device)
