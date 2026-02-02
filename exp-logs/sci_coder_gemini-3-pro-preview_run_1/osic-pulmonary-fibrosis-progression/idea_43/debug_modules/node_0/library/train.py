import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.data import get_dataloaders, get_test_loader
from library.model import BSHDAN
from library.utils import LaplaceLogLikelihoodLoss, calculate_metric, seed_everything


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        delta_week = batch["delta_week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # The model calculates fvc_pred and sigma_pred internally if delta_week/baseline_fvc are provided
        outputs = model(img_axial, img_coronal, tabular, delta_week, baseline_fvc)

        fvc_pred = outputs["fvc_pred"]
        sigma_pred = outputs["sigma_pred"]

        # Calculate loss
        loss = criterion(fvc_pred, sigma_pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()

    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            outputs = model(img_axial, img_coronal, tabular, delta_week, baseline_fvc)

            fvc_pred = outputs["fvc_pred"]
            sigma_pred = outputs["sigma_pred"]

            all_fvc_true.append(target.cpu().numpy())
            all_fvc_pred.append(fvc_pred.cpu().numpy())
            all_sigma_pred.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_fvc_true)
    y_pred = np.concatenate(all_fvc_pred)
    sigma_pred = np.concatenate(all_sigma_pred)

    # Calculate metric
    score = calculate_metric(y_true, y_pred, sigma_pred)
    return score


def train_model(debug=Config.DEBUG):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = BSHDAN()
    model.to(device)

    # DataLoaders
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX
    )
    criterion = LaplaceLogLikelihoodLoss()

    # Training State
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device}...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Score! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_submission(model=None):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device(Config.DEVICE)

    if model is None:
        # Load best model if not provided
        model = BSHDAN()
        model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

    model.eval()
    test_loader = get_test_loader()

    predictions = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_weeks = batch["patient_week"]  # List of strings

            # Forward pass
            outputs = model(img_axial, img_coronal, tabular, delta_week, baseline_fvc)

            fvc_pred = outputs["fvc_pred"].cpu().numpy()
            sigma_pred = outputs["sigma_pred"].cpu().numpy()

            # Aggregate results
            for pw, fvc, sigma in zip(patient_weeks, fvc_pred, sigma_pred):
                predictions.append(
                    {"Patient_Week": pw, "FVC": fvc, "Confidence": sigma}
                )

    # Create DataFrame
    sub_df = pd.DataFrame(predictions)

    # Ensure columns are in correct order
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(sub_df.head())


if __name__ == "__main__":
    # This block is strictly for local testing if run directly,
    # but the task requires no main block in the response.
    # The functions above are designed to be imported.
    pass
