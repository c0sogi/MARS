import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.train import Runner
from library.utils import seed_everything, MaskedL1Loss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration Override
    # ==========================================
    # Override Config for optimal convergence.
    # Increased to 80 epochs to ensure hybrid architecture convergence (Cite solution_lesson_node_00039).
    Config.EPOCHS = 80
    Config.DEBUG = False  # Use full dataset to meet the strict score threshold

    print("Configuration configured for fast baseline execution.")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    # Initialize Runner (loads data and model)
    runner = Runner()

    # Execute Training
    runner.train()

    # ==========================================
    # 3. Final Validation & Metric Calculation
    # ==========================================
    print("\nStarting Final Validation and Failure Analysis...")

    # Load the best model checkpoint
    best_model_path = Config.MODEL_PATH
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    device = torch.device(Config.DEVICE)
    model = runner.model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare for accumulation
    val_loader = runner.val_loader

    total_abs_error = 0.0
    total_valid_points = 0

    # storage for failure analysis
    all_inputs = []
    all_preds = []
    all_targets = []
    all_u_out = []

    with torch.no_grad():
        for x, y, u_out in val_loader:
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Inference
            preds = model(x)

            # Calculate Metric (Masked MAE)
            # We calculate manually to ensure global mean precision
            mask = (u_out == 0).float()
            abs_diff = torch.abs(preds.view_as(y) - y)
            masked_error = abs_diff * mask

            total_abs_error += masked_error.sum().item()
            total_valid_points += mask.sum().item()

            # Store data for failure analysis (move to CPU)
            all_preds.append(preds.view(-1).cpu().numpy())
            all_targets.append(y.view(-1).cpu().numpy())
            all_inputs.append(x.view(-1, x.shape[-1]).cpu().numpy())
            all_u_out.append(u_out.view(-1).cpu().numpy())

    # Compute Final Metric
    final_metric = (
        total_abs_error / total_valid_points if total_valid_points > 0 else 0.0
    )

    # Print exactly as required
    print("Final Validation Metric:", final_metric)

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")

    # Concatenate all batches
    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets)
    flat_inputs = np.concatenate(all_inputs)
    flat_u_out = np.concatenate(all_u_out)

    # Filter for Inspiratory Phase (u_out == 0)
    # The metric is only based on this phase, so analysis should focus here.
    insp_mask = flat_u_out == 0

    if np.sum(insp_mask) > 0:
        # Calculate errors for inspiratory phase
        errors = np.abs(flat_preds[insp_mask] - flat_targets[insp_mask])
        features_insp = flat_inputs[insp_mask]

        # Define feature names based on library.data_loader.add_features order
        # 1. Base features
        feature_names = [
            "u_in",
            "u_out",
            "R",
            "C",
            "dt",
            "u_in_diff1",
            "area",
            "R_uin",
            "area_C",
        ]
        # 2. Lead features
        for i in range(1, Config.LEAD_STEPS + 1):
            feature_names.append(f"u_in_lead{i}")

        print("Correlation between Error Magnitude and Features (Inspiratory Phase):")

        # Calculate correlation for each feature
        n_features = features_insp.shape[1]
        for i in range(n_features):
            if i < len(feature_names):
                feat_name = feature_names[i]
            else:
                feat_name = f"Feature_{i}"

            feat_values = features_insp[:, i]

            # Check for constant features (std=0) to avoid NaN correlation
            if np.std(feat_values) < 1e-9:
                corr = 0.0
            else:
                corr = np.corrcoef(errors, feat_values)[0, 1]

            print(f"{feat_name}: {corr:.4f}")
    else:
        print("No inspiratory phase data found for analysis.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    threshold = 0.16391726930343686

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) meets threshold ({threshold}). Generating submission..."
        )
        runner.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
