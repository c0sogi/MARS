import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
import library.config as config
from library.utils import seed_everything, calculate_roc_auc, save_checkpoint
from library.dataset import get_dataloaders
from library.model import SwinTransformerGLU
from library.train import train_one_epoch, validate, inference


def run_pipeline():
    # 1. Setup and Reproducibility
    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    # We use load_cached_data=True to utilize pre-computed features.
    # We use the full dataset but limit epochs for the 'fast baseline' requirement.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    # Determine tabular dimension from the dataset
    tabular_dim = train_loader.dataset.tabular_data.shape[1]
    print(f"Tabular Input Dimension: {tabular_dim}")

    model = SwinTransformerGLU(tabular_input_dim=tabular_dim, pretrained=True)
    model.to(device)

    # 4. Training Configuration
    # Overriding config.EPOCHS (15) to 5 for a faster baseline execution
    n_epochs = 5
    print(f"Training for {n_epochs} epochs...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler setup
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{n_epochs} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"state_dict": model.state_dict()}, best_model_path)

    # 6. Final Validation Metric
    # Requirement: Print full precision
    print(f"Final Validation Metric: {best_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Get predictions on validation set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Calculate Error
    errors = np.abs(val_targets - val_probs)

    # Get Metadata
    val_df = val_loader.dataset.df.copy()

    if len(val_df) == len(errors):
        # Preprocess for correlation
        val_df["error"] = errors

        # Handle missing values for correlation
        val_df["age_approx"] = val_df["age_approx"].fillna(val_df["age_approx"].mean())
        val_df["sex"] = val_df["sex"].fillna("unknown")
        val_df["anatom_site_general_challenge"] = val_df[
            "anatom_site_general_challenge"
        ].fillna("unknown")

        # Encode categorical
        val_df["sex_code"] = val_df["sex"].astype("category").cat.codes
        val_df["site_code"] = (
            val_df["anatom_site_general_challenge"].astype("category").cat.codes
        )

        # Calculate Correlations
        corr_age = val_df["age_approx"].corr(val_df["error"])
        corr_sex = val_df["sex_code"].corr(val_df["error"])
        corr_site = val_df["site_code"].corr(val_df["error"])

        print(f"Correlation (Error vs Age): {corr_age}")
        print(f"Correlation (Error vs Sex): {corr_sex}")
        print(f"Correlation (Error vs Site): {corr_site}")
    else:
        print("Error: Validation dataframe length mismatch with predictions.")

    # 8. Submission Generation
    threshold = 0.874794288335701
    if best_auc > threshold:
        print(
            f"\nMetric ({best_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        test_preds = inference(model, test_loader, device)

        # Create Submission DataFrame
        test_df = test_loader.dataset.df
        submission = pd.DataFrame(
            {"image_name": test_df["image_name"], "target": test_preds}
        )

        # Save
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric ({best_auc}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
