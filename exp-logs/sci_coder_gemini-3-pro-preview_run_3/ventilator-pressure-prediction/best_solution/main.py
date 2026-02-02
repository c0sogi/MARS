import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import CWDHNet, MaskedMAELoss
from library.train import train_one_epoch, validate_one_epoch
from library.inference import generate_predictions


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # Restoring full epoch schedule (80) to ensure convergence (Cite solution_lesson_node_00054)
    # Config.EPOCHS is already 80 in library/config.py
    # Keeping Batch Size 128 for generalization (Cite solution_lesson_node_00018)
    Config.BATCH_SIZE = 128

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("Loading datasets...")
    # Load full datasets, utilizing cache if available
    train_x, train_y, val_x, val_y, test_x = prepare_datasets(load_cached_data=True)

    # Create DataLoaders
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)

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

    # ---------------------------------------------------------
    # 3. Model Setup
    # ---------------------------------------------------------
    print("Initializing CWDH-Net model...")
    model = CWDHNet().to(device)
    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # Required Output
    print(f"Final Validation Metric: {best_val_loss}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\nRunning Failure Analysis on Validation Set...")
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    all_errors = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            # Calculate absolute error per time step
            abs_errors = torch.abs(preds - targets)

            # Mask for inspiratory phase (u_out == 0)
            # u_out is at index 1 in Config.CONT_FEATURES
            u_out_idx = Config.CONT_FEATURES.index("u_out")
            u_out = inputs[:, :, u_out_idx]
            mask = u_out == 0

            # Flatten to analyze distribution across all valid time steps
            mask_flat = mask.view(-1).cpu().numpy()
            errors_flat = abs_errors.view(-1).cpu().numpy()
            inputs_flat = inputs.view(-1, inputs.shape[2]).cpu().numpy()

            # Filter only inspiratory phase
            valid_indices = mask_flat.astype(bool)
            if valid_indices.sum() > 0:
                all_errors.append(errors_flat[valid_indices])
                all_inputs.append(inputs_flat[valid_indices])

    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_inputs = np.concatenate(all_inputs)

        # Build DataFrame for correlation analysis
        analysis_df = pd.DataFrame(all_inputs, columns=Config.CONT_FEATURES)
        analysis_df["error_magnitude"] = all_errors

        # Calculate correlation
        correlations = analysis_df.corr()["error_magnitude"].sort_values(
            ascending=False
        )
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations.drop("error_magnitude"))
    else:
        print("No inspiratory phase data found for failure analysis.")

    # ---------------------------------------------------------
    # 6. Submission Logic
    # ---------------------------------------------------------
    THRESHOLD = 0.1642141044139862

    if best_val_loss < THRESHOLD:
        print(
            f"\nValidation metric {best_val_loss} is better than threshold {THRESHOLD}."
        )
        print("Generating submission file...")
        generate_predictions(batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"\nValidation metric {best_val_loss} did not meet threshold {THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
