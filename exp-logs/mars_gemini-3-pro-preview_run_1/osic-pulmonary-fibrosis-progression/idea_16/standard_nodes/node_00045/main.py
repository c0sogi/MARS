import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import VisuallyContextualizedNet
from library.train import LaplaceLogLikelihoodLoss, train_one_epoch, validate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load data with caching enabled to speed up execution
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = VisuallyContextualizedNet().to(device)

    # 4. Training Setup
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # 5. Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect validation predictions for analysis
    val_targets = []
    val_preds = []
    val_sigmas = []
    val_meta = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            target = batch["target"].to(device)

            # Inference
            output = model(img_axial, img_coronal, tabular, weeks)

            # Store results
            val_targets.append(target.cpu().numpy())
            val_preds.append(output["fvc"].cpu().numpy())
            val_sigmas.append(output["confidence"].cpu().numpy())

            # Store metadata for analysis: [Age_Norm, Sex_Code, Smoking_Code, Percent_Norm, Weeks]
            # tabular is (B, 4), weeks is (B)
            meta_batch = torch.cat([tabular, weeks.unsqueeze(1)], dim=1).cpu().numpy()
            val_meta.append(meta_batch)

    val_targets = np.concatenate(val_targets)
    val_preds = np.concatenate(val_preds)
    val_sigmas = np.concatenate(val_sigmas)
    val_meta = np.concatenate(val_meta)

    # Calculate Final Metric
    final_metric = score_function(val_targets, val_preds, val_sigmas)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    abs_errors = np.abs(val_targets - val_preds)
    feature_names = ["Age_Norm", "Sex_Code", "Smoking_Code", "Percent_Norm", "Weeks"]
    df_analysis = pd.DataFrame(val_meta, columns=feature_names)
    df_analysis["Abs_Error"] = abs_errors

    correlations = df_analysis.corr()["Abs_Error"].drop("Abs_Error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        results = []
        with torch.no_grad():
            for batch in test_loader:
                img_axial = batch["img_axial"].to(device)
                img_coronal = batch["img_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                weeks = batch["weeks"].to(device)
                patient_weeks = batch["patient_week"]

                output = model(img_axial, img_coronal, tabular, weeks)

                fvc_preds = output["fvc"].cpu().numpy()
                conf_preds = output["confidence"].cpu().numpy()

                # Clip confidence as per requirement
                conf_preds = np.maximum(conf_preds, Config.MIN_CONFIDENCE)

                for pw, fvc, conf in zip(patient_weeks, fvc_preds, conf_preds):
                    results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

        df_sub = pd.DataFrame(results)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
