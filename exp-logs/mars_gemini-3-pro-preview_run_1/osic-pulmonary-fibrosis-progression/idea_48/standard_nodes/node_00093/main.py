import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import SLHDAN
from library.train import (
    train_one_epoch,
    validate,
    build_baseline_lookup,
    get_baseline_tensors,
    generate_submission,
)


def main():
    # 1. Configuration & Setup
    # Override Config for fast baseline execution
    Config.EPOCHS = 15
    Config.PATIENCE = 5

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Build lookups for baseline anchors (required for parametric model)
    train_lookup = build_baseline_lookup(train_loader.dataset)
    val_lookup = build_baseline_lookup(val_loader.dataset)

    # 3. Model Initialization
    print("Initializing model...")
    model = SLHDAN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss().to(device)

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_lookup
        )

        # Validate
        val_metric = validate(model, val_loader, criterion, device, val_lookup)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
        )

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Final Validation & Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_preds_fvc = []
    val_targets = []
    val_errors = []
    val_weeks = []
    val_meta_rows = []

    running_metric = 0.0

    # Detailed validation pass
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["image_ax"].to(device)
            img_cor = batch["image_cor"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            current_weeks = batch["weeks"].to(device).unsqueeze(1)
            patient_ids = batch["patient_id"]

            base_fvc, base_week = get_baseline_tensors(patient_ids, val_lookup, device)

            preds = model(
                img_ax,
                img_cor,
                tabular,
                base_fvc=base_fvc,
                base_week=base_week,
                current_week=current_weeks,
            )

            # Metric calculation
            metric = calculate_metric(preds, targets)
            running_metric += metric * img_ax.size(0)

            # Collect data for analysis
            pred_fvc = preds[:, 0]
            true_fvc = targets.view(-1)

            val_preds_fvc.extend(pred_fvc.tolist())
            val_targets.extend(true_fvc.tolist())
            val_errors.extend(torch.abs(true_fvc - pred_fvc).tolist())
            val_weeks.extend(batch["weeks"].view(-1).tolist())

            # Extract metadata features for correlation
            # Tabular tensor: [Age, Sex, Smoking, Percent]
            # We need to reconstruct or fetch from dataframe, but tensor is handy
            tab_np = tabular.cpu().numpy()
            for i in range(len(patient_ids)):
                val_meta_rows.append(
                    {
                        "Age": float(tab_np[i, 0]),
                        "Sex": float(tab_np[i, 1]),
                        "Smoking": float(tab_np[i, 2]),
                        "Percent": float(tab_np[i, 3]),
                    }
                )

    final_metric = running_metric / len(val_loader.dataset)
    print(f"Final Validation Metric: {final_metric}")

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(val_meta_rows)
    analysis_df["Weeks"] = val_weeks
    analysis_df["Error"] = val_errors
    analysis_df["Target_FVC"] = val_targets

    # Calculate Correlations
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print("\nCorrelation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = -6.510164260864258
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold}). Generating submission..."
        )

        sub_df = generate_submission(model, test_loader, device)

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
