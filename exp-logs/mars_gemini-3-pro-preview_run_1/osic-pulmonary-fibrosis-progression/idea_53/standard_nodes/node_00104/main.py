import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import NSLHN
from library.train import train_epoch, validate
from library.predict import inference


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.setup()

    # Configure for a fast baseline execution
    # The dataset is small (~1k samples), so we use full data but limited epochs
    Config.EPOCHS = 12
    Config.BATCH_SIZE = 32

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Uses load_cached_data=True implicitly via the library implementation
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = NSLHN().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss function with training clipping disabled to see gradients from large errors
    criterion = LaplaceLogLikelihoodLoss(
        clip_sigma=Config.CONFIDENCE_CLIP,
        clip_error=Config.MAX_ERROR,
        apply_error_clip_in_train=False,
    )

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validation Step
        val_score = validate(val_loader, model, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        # Simple progress print
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

    print("Training complete.")

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for analysis...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    model.eval()

    val_preds = []
    val_sigmas = []
    val_targets = []

    # Features for failure analysis
    # Tabular is: [pct_norm, age_norm, sex_val, is_ex, is_never, is_cur]
    feature_data = {
        "Percent": [],
        "Age": [],
        "Sex": [],
        "Smoking_Ex": [],
        "Smoking_Never": [],
        "Smoking_Cur": [],
        "Relative_Week": [],
        "Baseline_FVC": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["relative_week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Inference
            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

            # Collect results
            val_preds.extend(pred_fvc.cpu().numpy())
            val_sigmas.extend(pred_sigma.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

            # Collect features
            tab_np = tabular.cpu().numpy()
            feature_data["Percent"].extend(tab_np[:, 0])
            feature_data["Age"].extend(tab_np[:, 1])
            feature_data["Sex"].extend(tab_np[:, 2])
            feature_data["Smoking_Ex"].extend(tab_np[:, 3])
            feature_data["Smoking_Never"].extend(tab_np[:, 4])
            feature_data["Smoking_Cur"].extend(tab_np[:, 5])
            feature_data["Relative_Week"].extend(rel_week.cpu().numpy())
            feature_data["Baseline_FVC"].extend(base_fvc.cpu().numpy())

    # Calculate Final Metric
    val_preds_np = np.array(val_preds)
    val_sigmas_np = np.array(val_sigmas)
    val_targets_np = np.array(val_targets)

    final_metric = calculate_metric(val_preds_np, val_sigmas_np, val_targets_np)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    abs_errors = np.abs(val_targets_np - val_preds_np)

    # Create DataFrame
    analysis_df = pd.DataFrame(feature_data)
    analysis_df["Error_Magnitude"] = abs_errors

    print("\n--- Failure Analysis: Correlation with Error Magnitude ---")
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print(correlations)
    print("----------------------------------------------------------")

    # 7. Conditional Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        inference(
            model_path=Config.MODEL_SAVE_PATH,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
