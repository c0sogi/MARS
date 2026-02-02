import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import seed_everything, AverageMeter, LaplaceLogLikelihood
from library.data import get_dataloaders
from library.model import TSCPNet, negative_laplace_log_likelihood


def train_one_epoch(model, loader, optimizer, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move batch to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["base_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass: Predict parameters
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Parametric Inference for Training
        # FVC = Base + alpha * weeks
        fvc_pred = base_fvc + alpha * weeks

        # Confidence = sigma_base + sigma_growth * |weeks|
        sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

        # Calculate Loss
        # negative_laplace_log_likelihood handles the clipping of sigma (min 70) and delta (max 1000)
        loss = negative_laplace_log_likelihood(target, fvc_pred, sigma_pred)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def validate_one_epoch(model, loader, device):
    """
    Handles the validation loop for a single epoch.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Parametric Inference
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Calculate Metric (Higher is better)
            score = LaplaceLogLikelihood(target, fvc_pred, sigma_pred)
            scores.update(score, img_ax.size(0))

    return scores.avg


def run_training(epochs=30, batch_size=16, debug=False, patience=8, seed=42):
    """
    Orchestrates the training process including setup, loop, and early stopping.
    """
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cache=True, debug=debug
    )

    # Initialize Model
    model = TSCPNet().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training State
    best_score = -float("inf")
    best_model_path = "./working/best_model.pth"
    patience_counter = 0

    # Ensure working directory exists
    os.makedirs("./working", exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate_one_epoch(model, val_loader, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Early Stopping and Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"New best model saved. Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_model_path


def generate_submission(model_path, batch_size=16):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cache=True, debug=False
    )

    # Load Model
    model = TSCPNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model path {model_path} does not exist. Using random weights.")

    model.eval()

    all_fvc_preds = []
    all_sigma_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            # Predict parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate predictions
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Collect results (move to CPU numpy)
            all_fvc_preds.append(fvc_pred.cpu().numpy())
            all_sigma_preds.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    all_fvc_preds = np.concatenate(all_fvc_preds)
    all_sigma_preds = np.concatenate(all_sigma_preds)

    # Load test metadata to align with Patient_Week IDs
    # The test_loader iterates sequentially over metadata/test.csv
    test_df = pd.read_csv("./metadata/test.csv")

    if len(test_df) != len(all_fvc_preds):
        print(
            f"Error: Mismatch between metadata rows ({len(test_df)}) and predictions ({len(all_fvc_preds)})"
        )

    # Assign predictions
    test_df["FVC"] = all_fvc_preds
    test_df["Confidence"] = all_sigma_preds

    # Clip confidence at 70ml as per metric requirement
    test_df["Confidence"] = test_df["Confidence"].clip(lower=70)

    # Prepare submission dataframe
    submission = test_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to file
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
