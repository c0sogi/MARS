import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import SLHDAN
from library.utils import laplace_log_likelihood_metric


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    debug=False,
):
    """
    Trains the SLH-DAN model with Early Stopping and saves the best checkpoint.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for dataloaders.
        learning_rate (float): Initial learning rate.
        patience (int): Early stopping patience.
        debug (bool): If True, limits the number of batches for quick debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")

    # 2. Data Loading
    # Note: load_cached_data=True is passed to leverage the caching mechanism in library/data.py
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=Config.NUM_WORKERS, load_cached_data=True
    )

    # Determine tabular dimension dynamically from a batch
    sample_batch = next(iter(train_loader))
    tab_dim = sample_batch["tabular"].shape[1]
    print(f"Dynamic Tabular Dimension: {tab_dim}")

    # 3. Model Initialization
    model = SLHDAN(tabular_input_dim=tab_dim).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    # Ensure working directory exists for checkpoints
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        # --- Training Phase ---
        for i, batch in enumerate(train_loader):
            if debug and i > 5:
                break

            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            week = batch["week"].to(device)
            baseline_week = batch["baseline_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target_fvc = batch["target"].to(device)

            optimizer.zero_grad()

            # Forward Pass: Predict parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

            # Parametric Inference
            delta_t = week - baseline_week
            fvc_pred = baseline_fvc + alpha * delta_t

            # Confidence Inference
            confidence = sigma_base + sigma_growth * torch.abs(delta_t)

            # Loss Calculation (Loss = -Metric)
            # Metric returns negative value (higher is better), so we negate it to minimize loss
            score = laplace_log_likelihood_metric(target_fvc, fvc_pred, confidence)
            loss = -score

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_scores = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if debug and i > 5:
                    break

                img_ax = batch["image_axial"].to(device)
                img_cor = batch["image_coronal"].to(device)
                tab = batch["tabular"].to(device)

                week = batch["week"].to(device)
                baseline_week = batch["baseline_week"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                target_fvc = batch["target"].to(device)

                alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

                delta_t = week - baseline_week
                fvc_pred = baseline_fvc + alpha * delta_t
                confidence = sigma_base + sigma_growth * torch.abs(delta_t)

                # Metric calculation
                score = laplace_log_likelihood_metric(target_fvc, fvc_pred, confidence)
                val_scores.append(score.item())

        # Aggregation
        avg_train_loss = np.mean(train_losses)
        avg_val_score = np.mean(val_scores)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val Score: {avg_val_score}"
        )

        # --- Early Stopping ---
        if avg_val_score > best_metric:
            best_metric = avg_val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with score: {best_metric}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_metric


def predict_and_submit(batch_size=Config.BATCH_SIZE):
    """
    Loads the best model, generates predictions for the test set,
    and saves the submission file.
    """
    device = Config.DEVICE
    seed_everything(Config.SEED)

    print("Generating submission...")

    # 1. Load Data
    # We re-fetch loaders (caching ensures this is fast)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, load_cached_data=True)

    # Determine input dim again (consistent with training)
    sample_batch = next(iter(test_loader))
    tab_dim = sample_batch["tabular"].shape[1]

    # 2. Load Model
    model = SLHDAN(tabular_input_dim=tab_dim).to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    results = []

    # 3. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            week = batch["week"].to(device)
            baseline_week = batch["baseline_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]

            # Predict parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

            # Calculate FVC and Confidence
            delta_t = week - baseline_week
            fvc_pred = baseline_fvc + alpha * delta_t
            confidence = sigma_base + sigma_growth * torch.abs(delta_t)

            # Clip confidence (Metric logic does this, but we need explicit values for CSV)
            confidence = torch.clamp(confidence, min=Config.MIN_CONFIDENCE)

            # Move to CPU for storage
            fvc_pred_np = fvc_pred.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            week_np = week.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                w = int(week_np[i])
                patient_week = f"{pid}_{w}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": fvc_pred_np[i],
                        "Confidence": confidence_np[i],
                    }
                )

    # 4. Format Submission
    submission_df = pd.DataFrame(results)

    # Load sample submission to ensure correct row order and completeness
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Merge predictions onto sample submission structure
    # This ensures we have exactly the rows required by the competition
    final_sub = pd.merge(
        sample_sub[["Patient_Week"]], submission_df, on="Patient_Week", how="left"
    )

    # Fill missing values (if any alignment issues occurred) with defaults
    final_sub["FVC"] = final_sub["FVC"].fillna(2000)
    final_sub["Confidence"] = final_sub["Confidence"].fillna(100)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_experiment(debug=False):
    """
    Orchestrates the full training and submission pipeline.
    """
    print("Starting Experiment...")

    # Train
    best_score = train_model(debug=debug)
    print(f"Training completed. Best Validation Score: {best_score}")

    # Submit
    predict_and_submit()
    print("Experiment completed successfully.")
