import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import library modules
from library.config import Config
from library.dataset import LungDataset
from library.model import DPSDAN
from library.loss import LaplaceLogLikelihoodLoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tab_dense = batch["tab_dense"].to(device)

        baseline_fvc = batch["baseline_fvc"].to(device)
        delta_week = batch["delta_week"].to(device)
        target_fvc = batch["target_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        alpha, sigma_base, sigma_growth = model(img_axial, img_coronal, tab_dense)

        # Compute loss
        loss = loss_fn(
            alpha, sigma_base, sigma_growth, baseline_fvc, delta_week, target_fvc
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_axial.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes the competition metric (higher is better).
    """
    model.eval()
    running_metric = 0.0

    # Constants for metric calculation
    sqrt_2 = torch.tensor(np.sqrt(2), device=device)
    max_error = Config.MAX_ERROR
    min_sigma = Config.MIN_SIGMA

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tab_dense = batch["tab_dense"].to(device)

            baseline_fvc = batch["baseline_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            target_fvc = batch["target_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_axial, img_coronal, tab_dense)

            # Reconstruct Predictions
            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            # Calculate Metric
            # 1. Clipped Absolute Error
            abs_error = torch.abs(target_fvc - pred_fvc)
            delta = torch.clamp(abs_error, max=max_error)

            # 2. Clipped Confidence
            sigma_clipped = torch.clamp(pred_sigma, min=min_sigma)

            # 3. Laplace Log Likelihood Metric
            # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
            term1 = (sqrt_2 * delta) / sigma_clipped
            term2 = torch.log(sqrt_2 * sigma_clipped)
            batch_metric = -(term1 + term2)

            running_metric += batch_metric.sum().item()

    avg_metric = running_metric / len(loader.dataset)
    return avg_metric


def predict_and_submit(model, device):
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
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tab_dense = batch["tab_dense"].to(device)

            baseline_fvc = batch["baseline_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)

            # Identifiers
            patient_week_ids = batch["patient_week_id"]  # List of strings

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_axial, img_coronal, tab_dense)

            # Calculate Predictions
            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()

            for i, pw_id in enumerate(patient_week_ids):
                # Apply confidence clipping for submission as per metric definition logic
                # Though the metric clips at 70, the submission file just asks for confidence.
                # However, it's safer to provide the raw confidence or the clipped one.
                # The prompt says: "confidence values are clipped at 70 ml to reflect... uncertainty"
                # Usually, we submit the raw predicted sigma, and the evaluation metric clips it.
                # But to be safe and consistent with our optimization, we can clip it or leave it.
                # Given standard practice, we submit the value our model thinks is sigma.
                # Our model's sigma is strictly positive (Softplus).

                results.append(
                    {
                        "Patient_Week": pw_id,
                        "FVC": pred_fvc_np[i],
                        "Confidence": pred_sigma_np[i],
                    }
                )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns are correct
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


def run_training():
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Preparation
    print("Initializing Datasets...")
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")

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

    # 2. Model Initialization
    print("Initializing DP-SDAN Model...")
    model = DPSDAN().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    loss_fn = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! (Metric: {val_metric:.6f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation Metric: {best_metric:.6f}")

    # 5. Inference
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    predict_and_submit(model, device)


if __name__ == "__main__":
    run_training()
