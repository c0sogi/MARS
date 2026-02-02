import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import OSICDataset, get_transforms
from library.model import TMIGN, LaplaceLogLikelihoodLoss, train_one_epoch, validate


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    patience=Config.PATIENCE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    debug=False,
):
    """
    Orchestrates the training process with Early Stopping and Metric Tracking.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for dataloaders.
        patience (int): Early stopping patience.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        debug (bool): If True, uses a small subset of data for quick testing.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if debug:
        print("Debug mode: Using subset of data.")
        train_df = train_df.head(50)
        val_df = val_df.head(20)

    train_ds = OSICDataset(
        train_df, Config.CACHE_DIR, mode="train", transform=get_transforms("train")
    )
    val_ds = OSICDataset(
        val_df, Config.CACHE_DIR, mode="val", transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model and Components
    model = TMIGN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 3. Training Loop
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Metric: {val_metric}"
        )

        # Early Stopping & Checkpointing
        # Metric is negative Laplace Log Likelihood, higher is better
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Metric: {best_metric}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")


def generate_submission(batch_size=Config.BATCH_SIZE):
    """
    Generates submission file using the best trained model.

    Args:
        batch_size (int): Batch size for inference.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Model checkpoint not found at {best_model_path}")
        return

    model = TMIGN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 2. Prepare Test Data
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    # Rename columns to match Dataset expectation if needed,
    # though OSICDataset handles 'Baseline_' prefixes via mapping if implemented,
    # or we can rename here to be safe as per library.data.get_test_dataloader logic
    rename_map = {
        "Baseline_Age": "Age",
        "Baseline_Sex": "Sex",
        "Baseline_SmokingStatus": "SmokingStatus",
        "Baseline_Percent": "Percent",
    }
    test_df_renamed = test_df.rename(columns=rename_map)

    test_ds = OSICDataset(
        test_df_renamed, Config.CACHE_DIR, mode="test", transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            # Metadata for reconstruction
            target_weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)
            patient_ids = batch["patient_id"]

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Trajectory Calculation
            dt = target_weeks - base_week
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()
            target_weeks = target_weeks.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(target_weeks[i])
                patient_week = f"{pid}_{wk}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": pred_fvc[i],
                        "Confidence": pred_sigma[i],
                    }
                )

    # 3. Save Submission
    sub_df = pd.DataFrame(results)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
