import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import prepare_data
from library.model import CASDAN
from library.train import run_training
from library.predict import inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    # The dataset is small, but we limit epochs to ensure completion well within 2 hours.
    Config.EPOCHS = 20
    Config.PATIENCE = 5

    seed_everything(Config.SEED)

    print("==========================================")
    print(" ORCHESTRATION: CAS-DAN PIPELINE")
    print("==========================================")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n[Step 1/4] Starting Model Training...")
    # run_training handles data loading, model init, training loop, and saving best model
    run_training()

    # ==========================================
    # 3. Validation Phase
    # ==========================================
    print("\n[Step 2/4] Performing Validation...")

    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # load_cached_data=True ensures we use the data processed during training
    val_dataset = prepare_data("val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = CASDAN().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Inference on Validation Set
    val_preds = []
    val_sigmas = []
    val_targets = []
    val_weeks = []

    with torch.no_grad():
        for batch in val_loader:
            ax = batch["axial"].to(device)
            cor = batch["coronal"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["week"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            # Reconstruct Baseline FVC from normalized tabular data
            # Tabular structure: Age(0), Sex(1), Smk(2,3,4), Percent(5), BaseFVC(6)
            # Normalization was: (x - 2500) / 1000
            base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0

            # Predict FVC and Confidence
            fvc_pred = base_fvc_rec + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Collect results
            val_preds.extend(fvc_pred.cpu().numpy())
            val_sigmas.extend(sigma_pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
            val_weeks.extend(weeks.cpu().numpy())

    # Compute Metric
    final_metric = get_score(val_targets, val_preds, val_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n[Step 3/4] Failure Analysis...")

    # Construct Analysis DataFrame
    # val_dataset.df is aligned with the loader because shuffle=False
    df_analysis = val_dataset.df.copy()

    # Add predictions and errors
    df_analysis["Pred_FVC"] = val_preds
    df_analysis["Pred_Sigma"] = val_sigmas
    df_analysis["Abs_Error"] = np.abs(np.array(val_targets) - np.array(val_preds))

    # Prepare features for correlation analysis
    # Map categorical variables to numeric
    sex_map = {"Male": 0, "Female": 1}
    smk_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    df_analysis["Sex_Num"] = df_analysis["Sex"].map(sex_map).fillna(0)
    df_analysis["Smk_Num"] = df_analysis["SmokingStatus"].map(smk_map).fillna(0)

    # Select features for correlation
    features = ["Age", "Percent", "Weeks", "Sex_Num", "Smk_Num", "Abs_Error"]
    corr_matrix = df_analysis[features].corr()

    print("Correlation between Input Features and Absolute Error:")
    print(corr_matrix["Abs_Error"].drop("Abs_Error").sort_values(ascending=False))

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n[Step 4/4] Submission Generation...")

    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"Validation Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        inference()
    else:
        print(
            f"Validation Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
