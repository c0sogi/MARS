import os
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything, get_device, compute_metric
from library.dataset import get_dataloaders
from library.model import BCSLNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()

    # 2. Train
    # We limit epochs to 10 for a fast baseline execution.
    # The dataset is small enough that this runs quickly.
    print("Starting training...")
    _ = run_training(
        epochs=10, batch_size=16, patience=5, save_path="./working/best_model.pth"
    )

    # 3. Validation & Failure Analysis
    print("Running validation inference...")

    # Load Data
    _, val_loader, _ = get_dataloaders(batch_size=16)

    # Load Best Model
    model = BCSLNet().to(device)
    model_path = "./working/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using random weights.")

    model.eval()

    all_targets = []
    all_preds = []
    all_sigmas = []
    all_metas = []

    # Inference loop (No gradients for speed)
    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

            all_targets.append(target.cpu())
            all_preds.append(fvc_pred.cpu())
            all_sigmas.append(sigma_pred.cpu())

            # Store tabular data for failure analysis
            # Tabular columns: [Age, Sex, Smk_Ex, Smk_Never, Smk_Curr, Percent, Base_FVC_Scaled]
            all_metas.append(tabular.cpu())

    # Concatenate results
    targets = torch.cat(all_targets)
    preds = torch.cat(all_preds)
    sigmas = torch.cat(all_sigmas)
    metas = torch.cat(all_metas)

    # Compute Final Metric
    # Note: compute_metric returns a tensor, we use .item() to get the float
    final_metric = compute_metric(targets, preds, sigmas).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate Absolute Error
    errors = torch.abs(targets - preds).numpy()

    # Extract features (using normalized values is sufficient for correlation)
    # 0: Age, 5: Percent, 6: Base_FVC_Scaled
    age = metas[:, 0].numpy()
    percent = metas[:, 5].numpy()
    base_fvc_feat = metas[:, 6].numpy()

    analysis_df = pd.DataFrame(
        {"Abs_Error": errors, "Age": age, "Percent": percent, "Base_FVC": base_fvc_feat}
    )

    correlations = analysis_df.corr()["Abs_Error"].drop("Abs_Error")
    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 4. Submission
    # Threshold from instructions
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(f"\nValidation Metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission(
            model_path="./working/best_model.pth",
            output_path="./submission/submission.csv",
            batch_size=16,
        )
    else:
        print(
            f"\nValidation Metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
