import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import from provided libraries
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import LungDataset, get_transforms
from library.model import NSHDAN, train_one_epoch, validate, get_baseline_weeks


def train_model(epochs=30, batch_size=32, lr=1e-4, patience=8, limit_size=None):
    """
    Main training loop for the NSH-DAN model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        limit_size (int, optional): If provided, limits dataset size for debugging.

    Returns:
        float: The best validation loss achieved.
    """
    # Set seeds for reproducibility
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Data Loaders
    train_dataset = LungDataset(
        mode="train", transform=get_transforms("train"), limit_size=limit_size
    )
    val_dataset = LungDataset(
        mode="val", transform=get_transforms("val"), limit_size=limit_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Prepare Baseline Maps for Delta T calculation
    # We combine train and val maps to ensure coverage if needed, though split is strict
    train_base_map = get_baseline_weeks("train")
    val_base_map = get_baseline_weeks("val")
    full_base_map = {**train_base_map, **val_base_map}

    # Initialize Model, Loss, Optimizer, and Scheduler
    model = NSHDAN().to(device)
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Training State
    best_loss = float("inf")
    patience_counter = 0
    save_dir = "./working"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Execute Training Step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, full_base_map
        )

        # Execute Validation Step
        val_loss = validate(model, val_loader, criterion, device, full_base_map)

        # Update Learning Rate
        scheduler.step()

        # Log Metrics (Full Precision for Validation Loss)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved! New Best Val Loss: {best_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_loss


def generate_submission(batch_size=32):
    """
    Generates the submission file using the best trained model.

    Args:
        batch_size (int): Batch size for inference.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = NSHDAN().to(device)
    model_path = "./working/best_model.pth"

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Cannot generate submission.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize Test Data Loader
    test_dataset = LungDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Baseline Map for Test Set
    test_base_map = get_baseline_weeks("test")

    predictions = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            meta = batch["meta"].to(device)
            week = batch["week"].to(device)  # This is the Predict_Week
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            # Calculate delta_t (Time since baseline)
            base_weeks = []
            for pw in patient_weeks:
                # patient_week format is ID_Week. Split on last underscore to get ID.
                pid = pw.rsplit("_", 1)[0]
                base_weeks.append(test_base_map.get(pid, 0))

            base_weeks = torch.tensor(base_weeks, device=device, dtype=torch.float32)
            dt = week - base_weeks

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

            # Parametric Prediction
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Clip confidence as per metric requirement (min 70 ml)
            pred_sigma = torch.clamp(pred_sigma, min=70)

            # Collect results
            for i in range(len(patient_weeks)):
                predictions.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": pred_fvc[i].item(),
                        "Confidence": pred_sigma[i].item(),
                    }
                )

    # Save Submission to CSV
    sub_df = pd.DataFrame(predictions)
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with {len(sub_df)} rows.")
