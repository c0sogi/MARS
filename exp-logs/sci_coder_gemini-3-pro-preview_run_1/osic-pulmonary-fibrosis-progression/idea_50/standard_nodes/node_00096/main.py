import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import seed_everything
from library.train import train_model, generate_submission
from library.data import LungDataset, get_transforms
from library.model import NSHDAN, get_baseline_weeks


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Training
    # We use 15 epochs and batch size 32 for a fast but effective baseline
    # The dataset is small (~1100 samples), so this runs quickly.
    print("Starting training pipeline...")
    train_model(epochs=15, batch_size=32, lr=1e-4, patience=6)

    # 3. Validation and Failure Analysis
    print("Performing validation inference and failure analysis...")

    # Load the best model
    model = NSHDAN().to(device)
    model_path = "./working/best_model.pth"
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Validation Data
    val_dataset = LungDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_base_map = get_baseline_weeks("val")

    results = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            # Calculate delta_t
            base_weeks = []
            for pw in patient_weeks:
                pid = pw.rsplit("_", 1)[0]
                base_weeks.append(val_base_map.get(pid, 0))

            base_weeks = torch.tensor(base_weeks, device=device, dtype=torch.float32)
            dt = week - base_weeks

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

            # Parametric Prediction
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Metric Calculation
            sigma_clipped = torch.clamp(pred_sigma, min=70)
            abs_err = torch.abs(target - pred_fvc)
            delta = torch.clamp(abs_err, max=1000)

            # Metric formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
            sqrt_2 = np.sqrt(2)
            metric_batch = -(sqrt_2 * delta / sigma_clipped) - torch.log(
                sqrt_2 * sigma_clipped
            )

            # Store results for analysis
            target_np = target.cpu().numpy()
            pred_fvc_np = pred_fvc.cpu().numpy()
            metric_np = metric_batch.cpu().numpy()
            meta_np = meta.cpu().numpy()  # [Age, Sex, Smoking, Percent]

            for i in range(len(target_np)):
                results.append(
                    {
                        "FVC_True": target_np[i],
                        "FVC_Pred": pred_fvc_np[i],
                        "Metric": metric_np[i],
                        "Error": abs(target_np[i] - pred_fvc_np[i]),
                        "Age": meta_np[i, 0],
                        "Sex": meta_np[i, 1],
                        "Smoking": meta_np[i, 2],
                        "Percent": meta_np[i, 3],
                    }
                )

    # Compute Final Metric
    df_results = pd.DataFrame(results)
    final_metric = df_results["Metric"].mean()

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Features
    print("\nFailure Analysis (Correlation with Absolute Error):")
    # We check correlation of features with the absolute error magnitude
    features = ["Age", "Sex", "Smoking", "Percent"]
    correlations = df_results[["Error"] + features].corr()["Error"].drop("Error")
    print(correlations.sort_values(ascending=False))

    # 4. Submission Generation
    # Threshold check
    threshold = -6.510164260864258

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold ({threshold}). Generating submission..."
        )
        generate_submission(batch_size=32)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
