import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Add library path if needed (though usually current dir is in path)
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric_score
from library.train import run_training
from library.predict import run_inference
from library.data import get_dataloaders
from library.model import SCVRNet

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def analyze_performance(model, val_loader, device):
    """
    Runs inference on validation set, computes the final metric,
    and performs failure analysis (correlation of error with features).
    """
    model.eval()

    results = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Move data to device
            img_ax = data["img_ax"].to(device)
            img_cor = data["img_cor"].to(device)
            tabular = data["tabular"].to(device)

            target_fvc = data["target_fvc"].to(device)
            week_delta = data["week_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)

            # Forward pass
            params = model(img_ax, img_cor, tabular)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Reconstruct predictions
            fvc_pred = baseline_fvc + alpha * week_delta
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

            # Move to CPU for analysis
            fvc_pred_np = fvc_pred.cpu().numpy()
            sigma_pred_np = sigma_pred.cpu().numpy()
            target_fvc_np = target_fvc.cpu().numpy()

            # Extract tabular features for correlation analysis
            # tabular is [Age/100, Sex, Smoke, Percent/100]
            tabular_np = tabular.cpu().numpy()

            # Reconstruct week from delta and baseline (approximate for analysis)
            week_delta_np = week_delta.cpu().numpy()
            baseline_fvc_np = baseline_fvc.cpu().numpy()

            for i in range(len(fvc_pred_np)):
                results.append(
                    {
                        "FVC_Pred": fvc_pred_np[i],
                        "FVC_True": target_fvc_np[i],
                        "Sigma_Pred": sigma_pred_np[i],
                        "Error": np.abs(fvc_pred_np[i] - target_fvc_np[i]),
                        "Age": tabular_np[i, 0] * 100,
                        "Sex": tabular_np[i, 1],
                        "Smoking": tabular_np[i, 2],
                        "Percent": tabular_np[i, 3] * 100,
                        "Week_Delta": week_delta_np[i],
                        "Baseline_FVC": baseline_fvc_np[i],
                    }
                )

    df_results = pd.DataFrame(results)

    # Compute Final Metric on the whole set
    # We pass the arrays to the utility function
    final_metric = compute_metric_score(
        torch.tensor(df_results["FVC_Pred"].values),
        torch.tensor(df_results["FVC_True"].values),
        torch.tensor(df_results["Sigma_Pred"].values),
    )

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis (Correlation with Absolute Error):")
    features = ["Age", "Sex", "Smoking", "Percent", "Week_Delta", "Baseline_FVC"]
    correlations = df_results[features].corrwith(df_results["Error"])
    print(correlations)

    return final_metric


def main():
    # 1. Configure for Fast Baseline
    # We modify Config attributes before setup
    Config.EPOCHS = 10  # Limit epochs for speed
    Config.PATIENCE = 3  # Strict early stopping
    Config.BATCH_SIZE = 16  # Ensure fit on GPU
    Config.NUM_WORKERS = 2  # Moderate workers

    # Setup environment
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    print("Starting Training...")
    # run_training saves the best model to ./working/best_model.pth
    run_training(save_path="./working/best_model.pth")

    # 3. Load Best Model for Analysis
    print("\nLoading best model for analysis...")
    model = SCVRNet()
    model_path = "./working/best_model.pth"

    if not os.path.exists(model_path):
        print("Error: Model file not found. Training may have failed.")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Analyze Performance on Validation Set
    _, val_loader = get_dataloaders()
    val_metric = analyze_performance(model, val_loader, device)

    # 5. Conditional Submission
    THRESHOLD = -6.510164260864258

    if val_metric > THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        run_inference(
            model_path=model_path,
            output_path=submission_path,
            device_name=Config.DEVICE,
        )
    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
