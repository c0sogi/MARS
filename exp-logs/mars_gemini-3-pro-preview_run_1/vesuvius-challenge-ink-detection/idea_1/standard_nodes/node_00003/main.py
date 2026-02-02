import os
import torch
import pandas as pd
import numpy as np
from library.config import DEVICE
from library.utils import set_seed, fbeta_score
from library.data import get_dataloaders
from library.model import train_model, optimize_threshold
from library.inference import predict_fragment, generate_submission_file


def main():
    # 1. Ensure Reproducibility
    set_seed(42)

    # 2. Data Loading
    # Load data using cached files if available to speed up the process.
    # We use the default batch size defined in config.
    dataloaders = get_dataloaders(load_cached_data=True)

    # 3. Model Training
    # Train the DepthProjectionCNN. Increased epochs for deeper model convergence.
    print("Starting model training...")
    model = train_model(dataloaders, epochs=15, patience=4)

    # 4. Threshold Optimization
    # Find the best threshold on the validation set
    print("Optimizing threshold...")
    best_threshold = optimize_threshold(model, dataloaders["val"])

    # 5. Validation Metric & Failure Analysis
    print("Performing Failure Analysis and Final Validation...")
    model.eval()
    device = torch.device(DEVICE)

    all_preds = []
    all_targets = []

    # Lists to store data for failure analysis
    errors = []
    input_means = []
    input_stds = []

    with torch.no_grad():
        for inputs, targets in dataloaders["val"]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inference
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            # Store for global metric calculation
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            # --- Failure Analysis ---
            # Calculate Mean Absolute Error (MAE) per sample in the batch
            # preds, targets shape: (Batch, 1, H, W)
            batch_mae = torch.abs(preds - targets).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(batch_mae)

            # Calculate simple input features: Mean and Std of the 3D volume
            # inputs shape: (Batch, 65, H, W)
            batch_means = inputs.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = inputs.std(dim=(1, 2, 3)).cpu().numpy()

            input_means.extend(batch_means)
            input_stds.extend(batch_stds)

    # Calculate and print the Final Validation Metric
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    final_metric = fbeta_score(
        all_preds_tensor, all_targets_tensor, beta=0.5, threshold=best_threshold
    )
    print(f"Final Validation Metric: {final_metric}")

    # Calculate and print correlations for Failure Analysis
    analysis_df = pd.DataFrame(
        {"error": errors, "input_mean": input_means, "input_std": input_stds}
    )

    # Compute correlation of features with the error
    correlations = analysis_df.corr()["error"].drop("error")
    print("Failure Analysis - Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Inference and Submission
    # Only generate submission if the model meets the performance requirement
    target_metric = 0.2746392488479614
    if final_metric > target_metric:
        if "test" in dataloaders:
            print(
                f"Metric {final_metric:.6f} > {target_metric:.6f}. Generating predictions..."
            )
            # Generate full-fragment probability maps
            fragment_maps = predict_fragment(model, dataloaders["test"], device=device)

            # Save submission to the specified directory
            submission_path = "./submission/submission.csv"
            generate_submission_file(
                fragment_maps, best_threshold, submission_path=submission_path
            )
    else:
        print(
            f"Metric {final_metric:.6f} did not exceed {target_metric:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
