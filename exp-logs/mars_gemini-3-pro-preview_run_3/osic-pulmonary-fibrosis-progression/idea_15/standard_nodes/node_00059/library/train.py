import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihood
from library.data import get_dataloaders
from library.model import EADSNet


def criterion(mu, sigma, target):
    """
    Calculates the Negative Log Likelihood (NLL) for the Laplace distribution.
    Optimization Objective: Minimize L = (sqrt(2) * |y - mu|) / sigma + log(sigma)

    Args:
        mu (torch.Tensor): Predicted mean (scaled).
        sigma (torch.Tensor): Predicted uncertainty (scaled).
        target (torch.Tensor): True target value (scaled).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # sigma is already softplus-ed and has epsilon added in the model forward pass
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu.device))
    loss = (sqrt_2 * torch.abs(target - mu)) / sigma + torch.log(sigma)
    return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        time_rel = batch["time"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(images, tabular, time_rel)

        # Compute loss
        loss = criterion(mu, sigma, target)

        # Backward pass and optimization
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, loader, device, target_scaler):
    """
    Evaluates the model on the validation set using the official metric.
    Predictions are inverse-transformed to the original scale before metric calculation.
    """
    model.eval()
    all_targets = []
    all_mus = []
    all_sigmas = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            time_rel = batch["time"].to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(images, tabular, time_rel)

            # Move to CPU for inverse transform and collection
            mu_scaled = mu_scaled.cpu()
            sigma_scaled = sigma_scaled.cpu()

            # Inverse transform to original units (ml)
            mu_orig = target_scaler.inverse_transform(mu_scaled)
            sigma_orig = target_scaler.inverse_transform_sigma(sigma_scaled)

            # Get raw targets (original units) provided by the dataset
            # batch['raw_fvc'] is a tensor of floats
            raw_targets = batch["raw_fvc"]

            all_targets.append(raw_targets)
            all_mus.append(mu_orig.flatten())
            all_sigmas.append(sigma_orig.flatten())

    # Concatenate all batches
    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_mus)
    sigma = torch.cat(all_sigmas)

    # Calculate official metric
    # Note: LaplaceLogLikelihood handles clipping internally for the metric
    score = LaplaceLogLikelihood(y_true, y_pred, sigma)
    return score


def run_training():
    """
    Main execution function for training, evaluation, and submission generation.
    """
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    print(f"Initializing model ({Config.IDEA_NAME})...")
    device = torch.device(Config.DEVICE)
    model = EADSNet().to(device)

    # -------------------------------------------------------------------------
    # 3. Optimizer & Scheduler
    # -------------------------------------------------------------------------
    # Differential Learning Rates: Lower for backbone, higher for heads
    backbone_params = list(model.image_encoder.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": Config.LEARNING_RATE_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validation
        val_score = evaluate(model, val_loader, device, target_scaler)

        # Update Scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f} | Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        # Cite debug_lesson_7: Ensure clear logic when artifacts are purged; handle NaN to prevent silent failures.
        if np.isnan(val_score):
            print("Warning: Validation score is NaN. Skipping checkpoint.")
            patience_counter += 1
        elif val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Score! Model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                )
                break

    print(f"Training complete. Best Val Score: {best_score:.10f}")

    # Cite debug_lesson_8: Decouple artifact generation from model convergence.
    # Ensure a checkpoint exists for downstream consumers even if validation failed (NaNs).
    if not os.path.exists(Config.BEST_MODEL_PATH):
        # Check if current model is valid (no NaNs) before saving as fallback
        has_nans = any(torch.isnan(p).any() for p in model.parameters())
        if not has_nans:
            print(
                "Warning: Best model checkpoint not found. Saving current model as fallback."
            )
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            print("Error: Model weights contain NaNs. Cannot save fallback checkpoint.")

    # -------------------------------------------------------------------------
    # 5. Generate Submission
    # -------------------------------------------------------------------------
    print("Generating submission...")

    if os.path.exists(Config.BEST_MODEL_PATH):
        # Load best model (guaranteed to exist now)
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.BEST_MODEL_PATH}")

        model.eval()

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                time_rel = batch["time"].to(device)
                patient_weeks = batch["patient_week"]

                mu_scaled, sigma_scaled = model(images, tabular, time_rel)

                # Move to CPU
                mu_scaled = mu_scaled.cpu()
                sigma_scaled = sigma_scaled.cpu()

                # Inverse transform
                mu_orig = target_scaler.inverse_transform(mu_scaled).flatten().numpy()
                sigma_orig = (
                    target_scaler.inverse_transform_sigma(sigma_scaled)
                    .flatten()
                    .numpy()
                )

                # Post-process: Clip confidence at 70ml for submission
                sigma_final = np.maximum(sigma_orig, Config.SIGMA_CLIP)

                for pw, fvc, conf in zip(patient_weeks, mu_orig, sigma_final):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        # Save to CSV
        sub_df = pd.DataFrame(submission_rows)
        # Ensure correct column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Submission skipped due to lack of valid model checkpoint.")
