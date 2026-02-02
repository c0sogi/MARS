import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import gc

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import prepare_dataloaders
from library.model import SC_GI_BiLSTM
from library.trainer import Trainer, WeightedL1Loss
from library.inference import generate_submission


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    # We use the full dataset to ensure metric quality but limit epochs.
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 1024  # A100 allows larger batch size for speed
    Config.DEBUG = False  # Use full dataset

    # Ensure reproducible runs
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing SC-GI-BiLSTM Model...")
    model = SC_GI_BiLSTM().to(device)

    # ==========================================
    # 4. Training Setup
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler adapted to the reduced epoch count
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = WeightedL1Loss()

    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # ==========================================
    # 6. Final Validation & Metric
    # ==========================================
    # Load the best model saved during training for evaluation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model for validation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_mae = trainer.validate(val_loader)
    # Required output format
    print(f"Final Validation Metric: {final_mae}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis on Validation Set...")
    model.eval()

    all_features = []
    all_targets = []
    all_preds = []

    # Identify index of u_out to filter for inspiratory phase
    try:
        u_out_idx = Config.SELECTED_FEATURES.index("u_out")
    except ValueError:
        u_out_idx = -1

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            # We analyze only the inspiratory phase (u_out == 0) as per the metric
            u_out = x[:, :, u_out_idx]
            mask = u_out == 0

            if mask.sum() > 0:
                # Flatten sequences and apply mask
                x_flat = x[mask].cpu().numpy()
                y_flat = y[mask].cpu().numpy()
                p_flat = preds[mask].cpu().numpy()

                all_features.append(x_flat)
                all_targets.append(y_flat)
                all_preds.append(p_flat)

    if all_features:
        # Concatenate all batches
        X_analysis = np.concatenate(all_features, axis=0)
        y_analysis = np.concatenate(all_targets, axis=0)
        p_analysis = np.concatenate(all_preds, axis=0)

        # Calculate Absolute Error
        errors = np.abs(y_analysis - p_analysis)

        # Create DataFrame for correlation analysis
        df_analysis = pd.DataFrame(X_analysis, columns=Config.SELECTED_FEATURES)
        df_analysis["error_magnitude"] = errors

        # Compute correlation between features and error magnitude
        correlations = (
            df_analysis.corr()["error_magnitude"]
            .drop("error_magnitude")
            .sort_values(ascending=False)
        )

        print(
            "\nCorrelation between Error Magnitude and Input Features (Inspiratory Phase):"
        )
        print(correlations)

        # Cleanup
        del df_analysis, X_analysis, y_analysis, p_analysis, errors
        gc.collect()

    # ==========================================
    # 8. Submission Generation
    # ==========================================
    THRESHOLD = 0.19242813024255964

    if final_mae < THRESHOLD:
        print(
            f"\nValidation Metric {final_mae} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model_path=best_model_path, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
        )
    else:
        print(
            f"\nValidation Metric {final_mae} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
