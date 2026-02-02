import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.train import run_training
from library.predict import (
    generate_submission,
    load_ensemble_models,
    predict_with_tta_ensemble,
)
from library.dataset import get_dataloaders
from library.utils import seed_everything


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(42)
    device = Config.DEVICE
    print(f"Orchestration started on device: {device}")

    # 2. Training
    # We use Config.EPOCHS (1000) to ensure full convergence (Cite solution_lesson_node_00016)
    print("--- Starting Training Phase ---")
    run_training(epochs=Config.EPOCHS, debug=False)
    print("Training phase complete.")

    # 3. Validation & Failure Analysis
    print("--- Starting Validation & Analysis Phase ---")

    # Load validation data
    # get_dataloaders handles caching internally
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the trained ensemble
    try:
        models = load_ensemble_models(device)
    except RuntimeError as e:
        print(f"Critical Error: {e}")
        return

    val_rmse_list = []
    meta_stats = []

    # accumulators for global RMSE calculation
    all_preds_flat = []
    all_targets_flat = []

    print("Running inference on validation set...")

    # Disable gradients for inference to save memory and speed up
    with torch.no_grad():
        for batch in val_loader:
            noisy, clean, img_id = batch

            # noisy: (1, 1, H, W), clean: (1, 1, H, W)

            # 3.1 Extract features for Failure Analysis before moving to GPU
            # Convert to numpy for stats
            n_np = noisy.squeeze().numpy()

            stats = {
                "id": str(img_id[0]),
                "mean_intensity": float(np.mean(n_np)),
                "std_intensity": float(np.std(n_np)),
                "height": int(n_np.shape[0]),
                "width": int(n_np.shape[1]),
                "area": int(n_np.shape[0] * n_np.shape[1]),
            }
            meta_stats.append(stats)

            # 3.2 Inference
            # Move to device
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Use TTA Ensemble prediction (same as submission pipeline)
            pred = predict_with_tta_ensemble(models, noisy, device)

            # 3.3 Error Calculation
            # Per-image RMSE for correlation analysis
            # We calculate RMSE on the CPU to avoid accumulating GPU tensors
            pred_cpu = pred.cpu()
            clean_cpu = clean.cpu()

            mse_val = torch.mean((pred_cpu - clean_cpu) ** 2).item()
            rmse_val = np.sqrt(mse_val)
            val_rmse_list.append(rmse_val)

            # Accumulate flattened data for Global RMSE
            all_preds_flat.append(pred_cpu.flatten())
            all_targets_flat.append(clean_cpu.flatten())

    # 4. Compute Metrics

    # Global RMSE: sqrt(mean((all_preds - all_targets)^2))
    # Concatenate all pixels into one large 1D tensor
    full_preds = torch.cat(all_preds_flat)
    full_targets = torch.cat(all_targets_flat)

    global_mse = torch.mean((full_preds - full_targets) ** 2).item()
    global_rmse = np.sqrt(global_mse)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {global_rmse}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(meta_stats)
    df_analysis["rmse"] = val_rmse_list

    # Calculate correlations
    features = ["mean_intensity", "std_intensity", "height", "width", "area"]
    print("Correlation between Input Features and Error (RMSE):")
    for feat in features:
        if feat in df_analysis.columns:
            # Check for constant values to avoid NaN correlation
            if df_analysis[feat].std() > 0 and df_analysis["rmse"].std() > 0:
                corr = np.corrcoef(df_analysis[feat], df_analysis["rmse"])[0, 1]
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: N/A (Constant values)")

    # 6. Submission Generation
    # Threshold defined in task (Best known: 0.01187)
    THRESHOLD = 0.011870221132053216

    print("\n--- Submission Decision ---")
    print(f"Threshold: {THRESHOLD}")
    print(f"Achieved:  {global_rmse}")

    if global_rmse < THRESHOLD:
        print("Validation metric meets the threshold. Generating submission...")
        generate_submission()
    else:
        print("Validation metric does NOT meet the threshold. Skipping submission.")


if __name__ == "__main__":
    main()
