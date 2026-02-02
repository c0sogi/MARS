import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import TSCPNet, validate
from library.train import train_model, generate_predictions


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Train Model
    # Limiting epochs to 20 for a fast baseline execution as requested.
    # The dataset is small (~1k samples), so this will run very quickly on an A100.
    print("\n=== Starting Training ===")
    best_val_score = train_model(
        epochs=20,
        batch_size=Config.BATCH_SIZE,
        device_name=Config.DEVICE,
        checkpoint_dir=Config.CHECKPOINT_DIR,
    )

    # 3. Validation Evaluation
    print("\n=== Performing Validation ===")
    # Load the best model
    model = TSCPNet().to(device)
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get dataloaders
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # Calculate final metric using the provided validate function
    # This ensures consistency with the training loop metric
    final_metric = validate(model, val_loader, device)

    # Print required metric format (Full precision)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            meta = batch["meta"].to(device)
            week_diff = batch["week_diff"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            params = model(img_ax, img_cor, meta)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Predict
            pred_fvc = baseline_fvc + alpha * week_diff

            # Calculate Absolute Error
            # Move to CPU for analysis
            t_fvc_np = target.cpu().numpy()
            p_fvc_np = pred_fvc.cpu().numpy()

            abs_error = np.abs(t_fvc_np - p_fvc_np)

            # Extract features for correlation
            # meta: [Age_norm, Sex_M, Sex_F, Smoke_Ex, Smoke_Nev, Smoke_Cur, Baseline_Percent_norm, Baseline_FVC_norm]
            meta_np = meta.cpu().numpy()
            week_diff_np = week_diff.cpu().numpy()

            for i in range(len(t_fvc_np)):
                analysis_data.append(
                    {
                        "Abs_Error": abs_error[i],
                        "Week_Diff": np.abs(week_diff_np[i]),  # Magnitude of time delta
                        "Age": meta_np[i, 0],
                        "Sex_Male": meta_np[i, 1],
                        "Smoke_Cur": meta_np[i, 5],
                        "Baseline_Percent": meta_np[i, 6],
                        "Baseline_FVC": meta_np[i, 7],
                    }
                )

    # Create DataFrame and compute correlations
    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty:
        correlations = df_analysis.corr()["Abs_Error"].sort_values(ascending=False)
        print("Correlation between Absolute Error and Input Features:")
        print(correlations)
    else:
        print("No validation data available for analysis.")

    # 5. Submission Generation
    TARGET_THRESHOLD = -6.510164260864258

    print("\n=== Submission Check ===")
    print(f"Threshold: {TARGET_THRESHOLD}")
    print(f"Model Score: {final_metric}")

    if final_metric > TARGET_THRESHOLD:
        print("Score exceeds threshold. Generating submission...")
        generate_predictions(
            model_path=best_model_path,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device_name=Config.DEVICE,
        )
    else:
        print("Score does not exceed threshold. Submission skipped.")


if __name__ == "__main__":
    main()
