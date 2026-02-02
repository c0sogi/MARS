import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.train import train_model
from library.inference import generate_submission
from library.data import get_dataloaders
from library.model import PGBBNet


def run_failure_analysis(device):
    """
    Performs failure analysis on the validation set using the best saved model.
    Calculates correlation between Absolute Error and Input Features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Load Data
    _, val_loader = get_dataloaders(debug=Config.DEBUG)

    # 2. Load Model
    model = PGBBNet().to(device)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Best model not found. Skipping failure analysis.")
        return

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # 3. Collect Predictions and Features
    errors = []
    feats_age = []
    feats_sex = []
    feats_smoke = []
    feats_percent = []
    feats_week = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            # Inference
            preds = model(axial, coronal, tabular, meta)
            fvc_pred = preds[:, 0]

            # Calculate Absolute Error
            abs_error = torch.abs(target - fvc_pred).cpu().numpy()
            errors.extend(abs_error)

            # Extract Features
            # Tabular: [Age, Sex, Smoke, Percent]
            # Meta: [Rel_Week, Base_FVC]
            tab_np = tabular.cpu().numpy()
            meta_np = meta.cpu().numpy()

            feats_age.extend(tab_np[:, 0])
            feats_sex.extend(tab_np[:, 1])
            feats_smoke.extend(tab_np[:, 2])
            feats_percent.extend(tab_np[:, 3])
            feats_week.extend(meta_np[:, 0])

    # 4. Compute Correlations
    data = pd.DataFrame(
        {
            "AbsError": errors,
            "Age": feats_age,
            "Sex": feats_sex,
            "Smoking": feats_smoke,
            "Percent": feats_percent,
            "Week": feats_week,
        }
    )

    print(f"Analyzed {len(data)} validation samples.")
    print("Correlation between Absolute Error and Features:")

    features = ["Age", "Sex", "Smoking", "Percent", "Week"]
    for feat in features:
        if data[feat].nunique() > 1:
            corr, _ = pearsonr(data["AbsError"], data[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: N/A (Constant value)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Configuration Override
    # Limit epochs to ensure completion within time limit (54 mins)
    # The dataset is small, so 15 epochs is sufficient for convergence
    Config.EPOCHS = 15

    print("Starting PGBB-Net Pipeline...")

    # 3. Train Model
    # train_model returns the best validation loss (Negative Log Likelihood)
    best_loss = train_model()

    # Metric is negative loss
    final_metric = -best_loss

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    run_failure_analysis(device)

    # 5. Submission
    # Threshold defined in task description logic
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) meets threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
            output_file=Config.SUBMISSION_FILE,
        )
    else:
        print(
            f"\nMetric ({final_metric:.4f}) does not meet threshold ({THRESHOLD:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
