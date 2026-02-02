import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import the library correctly
sys.path.append(os.getcwd())

from library import config, utils, data, model, train


def main():
    # --- 1. Configuration Overrides ---
    # We override specific configurations to satisfy the "fast baseline" requirement
    # and to ensure the submission path matches the task description.
    config.EPOCHS = (
        30  # Increased to ensure convergence (Cite solution_lesson_node_00019)
    )
    config.BATCH_SIZE = 512  # Balanced batch size
    config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure the submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Set random seeds for reproducibility
    utils.seed_everything()

    print(f"Configuration Configured:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Submission Path: {config.SUBMISSION_PATH}")

    # --- 2. Training ---
    print("\n=== Starting Training ===")
    # Train the model using the library's training loop.
    # Note: train_model uses the default batch size (512) defined at import time in data.py,
    # which is acceptable. The epoch count will use the overridden config.EPOCHS.
    train.train_model()

    # --- 3. Validation & Failure Analysis ---
    print("\n=== Starting Validation & Failure Analysis ===")

    device = utils.get_device()

    # Initialize model and load the best weights saved during training
    net = model.PhysicsResidualModel().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    # Get the validation data loader
    # We explicitly pass the batch size here to utilize the overridden value (1024)
    _, val_loader, _ = data.get_data_loaders(
        load_cached_data=True, batch_size=config.BATCH_SIZE
    )

    total_ae = 0.0
    total_mask_sum = 0.0

    # Lists to store data for failure analysis
    error_list = []
    u_in_list = []
    time_list = []
    r_list = []
    c_list = []

    print("Computing validation metrics...")
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            x_cont = batch["x_cont"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Forward pass
            preds = net(x_cont)

            # Calculate Masked Absolute Error
            # We only care about the inspiratory phase (u_out == 0)
            mask = 1 - u_out
            abs_error = torch.abs(preds - y)
            masked_error = abs_error * mask

            # Accumulate sums for global MAE
            total_ae += masked_error.sum().item()
            total_mask_sum += mask.sum().item()

            # Collect feature and error data for failure analysis
            # We filter by mask to only analyze the relevant phase
            mask_bool = mask.bool()

            if mask_bool.any():
                # Extract relevant data points and move to CPU/Numpy
                error_list.append(abs_error[mask_bool].cpu().numpy())

                # x_cont features: 0=time_step, 1=u_in
                # Note: Indices depend on Config.CONTINUOUS_FEATURES order
                # time_step is at index 0, u_in at index 1
                time_list.append(x_cont[:, :, 0][mask_bool].cpu().numpy())
                u_in_list.append(x_cont[:, :, 1][mask_bool].cpu().numpy())

                # R is at index 3, C is at index 4 in new config
                r_list.append(x_cont[:, :, 3][mask_bool].cpu().numpy())
                c_list.append(x_cont[:, :, 4][mask_bool].cpu().numpy())

    # Calculate Final Metric
    final_metric = total_ae / (total_mask_sum + 1e-8)
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    if len(error_list) > 0:
        all_errors = np.concatenate(error_list)
        all_u_in = np.concatenate(u_in_list)
        all_time = np.concatenate(time_list)
        all_r = np.concatenate(r_list)
        all_c = np.concatenate(c_list)

        print("\nFailure Analysis - Correlation with Absolute Error:")
        # Calculate Pearson correlation coefficient
        corr_u_in = np.corrcoef(all_errors, all_u_in)[0, 1]
        corr_time = np.corrcoef(all_errors, all_time)[0, 1]
        corr_r = np.corrcoef(all_errors, all_r)[0, 1]
        corr_c = np.corrcoef(all_errors, all_c)[0, 1]

        print(f"  u_in:      {corr_u_in:.4f}")
        print(f"  time_step: {corr_time:.4f}")
        print(f"  R:         {corr_r:.4f}")
        print(f"  C:         {corr_c:.4f}")

    # --- 4. Submission ---
    # Check against the provided threshold
    THRESHOLD = 0.4283660650253296

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        # predict_and_submit uses config.SUBMISSION_PATH which we updated
        train.predict_and_submit()
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
