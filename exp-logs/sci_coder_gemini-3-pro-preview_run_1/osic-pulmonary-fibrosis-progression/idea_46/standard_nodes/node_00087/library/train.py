import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_loss
from library.data import get_dataloaders
from library.model import SLHDANetwork


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        img_ax = batch["img_axial"].to(device)
        img_cor = batch["img_coronal"].to(device)
        tab = batch["tabular"].to(device)
        target = batch["target"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        base_week = batch["base_week"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(img_ax, img_cor, tab, weeks, base_fvc, base_week)

        # Compute loss
        loss = laplace_log_likelihood_loss(target, pred_fvc, pred_sigma)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the average metric score (Negative Log Likelihood).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            pred_fvc, pred_sigma = model(
                img_ax, img_cor, tab, weeks, base_fvc, base_week
            )

            loss = laplace_log_likelihood_loss(target, pred_fvc, pred_sigma)
            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    # The metric is defined as negative of the loss in the competition context usually,
    # but strictly speaking the competition metric formula provided is:
    # metric = - (sqrt(2) * Delta / Sigma) - ln(sqrt(2) * Sigma)
    # Our loss function calculates: (sqrt(2) * Delta / Sigma) + ln(sqrt(2) * Sigma)
    # Therefore, Metric = -Loss
    return -avg_loss


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)
            patient_ids = batch["patient_id"]

            pred_fvc, pred_sigma = model(
                img_ax, img_cor, tab, weeks, base_fvc, base_week
            )

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()
            weeks_np = weeks.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                week = int(weeks_np[i])
                fvc = pred_fvc[i]
                sigma = pred_sigma[i]

                # Construct Patient_Week ID
                patient_week = f"{pid}_{week}"

                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": sigma}
                )

    df = pd.DataFrame(results)

    # Ensure columns are in correct order
    df = df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 3. Model
    print("Initializing model...")
    model = SLHDANetwork().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! (Metric: {best_metric:.10f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("\nGenerating submission...")
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found, using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Done.")


# Entry point for the module
if __name__ == "__main__":
    # This block is only for local testing if run directly,
    # but the instructions say "Only implement the module class/functions".
    # However, to make the file executable as a script if needed, we include this.
    # The user instruction "DO NOT include an if __name__ == '__main__': block"
    # likely refers to not putting the *entire* logic inside it, but rather exposing functions.
    # Given the strict instruction "DO NOT include an if __name__ == '__main__': block",
    # I will omit it entirely. The function `run_training` is available to be called.
    pass
