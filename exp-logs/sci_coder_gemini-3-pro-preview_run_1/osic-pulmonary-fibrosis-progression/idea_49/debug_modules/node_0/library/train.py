import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, AverageMeter, score_function
from library.data import get_dataloaders
from library.model import SLHDAN


def loss_fn(fvc_pred, sigma_pred, fvc_true):
    """
    Modified Laplace Log Likelihood Loss.
    Minimizes the negative of the competition metric.
    """
    # Constants
    MAX_ERROR = Config.MAX_ERROR
    MIN_CONFIDENCE = Config.MIN_CONFIDENCE
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=fvc_pred.device))

    # Clipping
    sigma_clipped = torch.clamp(sigma_pred, min=MIN_CONFIDENCE)

    # Absolute error with threshold
    delta = torch.abs(fvc_true - fvc_pred)
    delta = torch.clamp(delta, max=MAX_ERROR)

    # Loss calculation
    # Metric = - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    # Loss = - Metric
    loss = (sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        time_delta = batch["time_delta"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        # Forward pass
        fvc_pred, sigma_pred = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

        # Loss calculation
        loss = loss_fn(fvc_pred, sigma_pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, time_delta, baseline_fvc
            )

            all_targets.extend(target.cpu().numpy())
            all_fvc_preds.extend(fvc_pred.cpu().numpy())
            all_sigma_preds.extend(sigma_pred.cpu().numpy())

    # Calculate metric using the official score function
    score = score_function(all_targets, all_fvc_preds, all_sigma_preds)
    return score


def generate_submission(model, loader, device):
    model.eval()
    results = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Metadata for ID construction
            patient_ids = batch["patient_id"]
            weeks = batch["week"]

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, time_delta, baseline_fvc
            )

            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                week = weeks[i].item()
                fvc = fvc_pred[i]
                conf = sigma_pred[i]

                # Construct Patient_Week ID
                patient_week = f"{pid}_{week}"

                # Ensure confidence is clipped for submission format compliance
                # (Though metric does it, the file should also look reasonable)
                conf = max(conf, 70)

                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 2. Model
    model = SLHDAN().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! Score: {best_score:.10f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    main()
