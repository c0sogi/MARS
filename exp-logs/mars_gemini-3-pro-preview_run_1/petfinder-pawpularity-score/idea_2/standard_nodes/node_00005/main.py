import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_rmse,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import PawpularitySwinModel
from library.engine import train_one_epoch, validate, inference


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between absolute error and input features.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_features = []

    # We need to collect features corresponding to the batches
    # The loader returns features in the batch dictionary

    with torch.no_grad():
        for batch_data in val_loader:
            images = batch_data["image"].to(device)
            features = batch_data["features"].to(device)
            targets = batch_data["target"].to(device)

            outputs = model(images, features)
            outputs = outputs.view(-1)
            targets = targets.view(-1)

            # Convert to scale [0, 100]
            preds = torch.sigmoid(outputs) * 100.0
            targets_scaled = targets * 100.0

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets_scaled.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_features = np.array(all_features)

    # Calculate residuals (absolute error)
    errors = np.abs(all_targets - all_preds)

    # Create a DataFrame for correlation analysis
    feature_names = Config.feature_cols
    analysis_df = pd.DataFrame(all_features, columns=feature_names)
    analysis_df["Error"] = errors

    # Calculate correlation
    correlations = analysis_df.corr()["Error"].drop("Error")

    print("\n=== Failure Analysis: Correlation with Error Magnitude ===")
    print(correlations.sort_values(ascending=False).to_string())
    print("========================================================\n")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # Override Config for fast baseline execution
    # 5 epochs is usually enough for fine-tuning a pre-trained Swin Transformer on a small dataset
    Config.epochs = 5

    print(f"Device: {device}")
    print(f"Epochs: {Config.epochs}")
    print(f"Model: {Config.model_name}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model Initialization
    print("Initializing model...")
    model = PawpularitySwinModel(pretrained=True)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 5. Training Loop
    best_rmse = float("inf")

    print("Starting training...")
    for epoch in range(1, Config.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        # Save Best Model
        if val_rmse < best_rmse:
            print(
                f"RMSE improved from {best_rmse:.4f} to {val_rmse:.4f}. Saving model..."
            )
            best_rmse = val_rmse
            save_checkpoint(model, optimizer, epoch, val_loss, Config.model_save_path)

    print("Training complete.")

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    # Re-initialize model to ensure clean state or just load weights
    model = PawpularitySwinModel(
        pretrained=False
    )  # Pretrained=False is faster to init, weights will be loaded
    model.to(device)

    _, _ = load_checkpoint(Config.model_save_path, model, device=device)

    # Calculate Final Metric
    print("Calculating final validation metric...")
    _, final_rmse = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    # Threshold from requirements: 19.2663631439209
    THRESHOLD = 19.2663631439209

    if final_rmse < THRESHOLD:
        print(
            f"Validation RMSE ({final_rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        inference(model, test_loader, device, Config.submission_path)
    else:
        print(
            f"Validation RMSE ({final_rmse}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
