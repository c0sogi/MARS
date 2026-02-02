import os
import torch
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from library import config, utils, model, dataset, engine, inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    # We override the default configuration to ensure the script runs quickly
    # as a baseline verification.
    RUN_EPOCHS = 30  # Reduced from 1000 to ensure fast execution
    RUN_SEEDS = [42]  # Use a single seed instead of the full ensemble

    # Set global seed for reproducibility
    utils.set_seed(42)

    device = config.DEVICE
    print(f"Running on device: {device}")
    print(f"Fast Baseline Configuration: Epochs={RUN_EPOCHS}, Seeds={RUN_SEEDS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # load_cached=True utilizes the preprocessed .npz files in ./working if available
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=2,  # Reduced workers for lighter overhead
        load_cached=True,
    )

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    for seed in RUN_SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        utils.set_seed(seed)

        # Initialize Model
        net = model.WideBottleneckUNet(
            n_channels=config.IN_CHANNELS, n_classes=config.OUT_CHANNELS
        ).to(device)

        # Initialize Optimizer
        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

        # Initialize Scheduler
        # T_max matches the number of epochs for this run
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUN_EPOCHS)

        # Define checkpoint path
        save_path = config.get_checkpoint_path(seed)

        # Train
        engine.fit(
            model=net,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=RUN_EPOCHS,
            save_path=save_path,
        )

        # Clean up to save memory
        del net, optimizer, scheduler
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation Assessment & Failure Analysis ---")

    # Load the trained model (Seed 42)
    best_model_path = config.get_checkpoint_path(42)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    net = model.WideBottleneckUNet(
        n_channels=config.IN_CHANNELS, n_classes=config.OUT_CHANNELS
    ).to(device)
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.eval()

    # Variables for analysis
    val_rmse_list = []
    input_means = []
    input_stds = []

    total_sse = 0.0
    total_pixels = 0

    with torch.no_grad():
        for data in val_loader:
            # Unpack data (Val loader returns noisy, clean, meta)
            if len(data) == 3:
                noisy, clean, meta = data
            else:
                continue

            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward pass
            output = net(noisy)

            # Get original dimensions to unpad
            h_orig = meta["orig_h"].item()
            w_orig = meta["orig_w"].item()

            # Extract valid regions (CPU numpy arrays)
            pred_valid = output[0, 0, :h_orig, :w_orig].cpu().numpy()
            target_valid = clean[0, 0, :h_orig, :w_orig].cpu().numpy()
            input_valid = noisy[0, 0, :h_orig, :w_orig].cpu().numpy()

            # --- 1. Metric Calculation ---
            # Per-image SSE
            sse_img = np.sum((pred_valid - target_valid) ** 2)
            total_sse += sse_img
            total_pixels += h_orig * w_orig

            # Per-image RMSE for correlation
            mse_img = np.mean((pred_valid - target_valid) ** 2)
            rmse_img = np.sqrt(mse_img)
            val_rmse_list.append(rmse_img)

            # --- 2. Feature Extraction for Failure Analysis ---
            input_means.append(np.mean(input_valid))
            input_stds.append(np.std(input_valid))

    # Compute Final Metric
    final_metric = np.sqrt(total_sse / total_pixels) if total_pixels > 0 else 0.0

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_metric:.20f}")

    # Compute Correlations
    if len(val_rmse_list) > 1:
        corr_mean, _ = pearsonr(val_rmse_list, input_means)
        corr_std, _ = pearsonr(val_rmse_list, input_stds)

        print("-" * 30)
        print("Failure Analysis (Correlation with Error Magnitude)")
        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Std Dev): {corr_std:.4f}")
        print("-" * 30)
    else:
        print("Insufficient validation samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in task
    THRESHOLD = 0.011870221132053216

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric:.6f} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Run inference using the trained seed(s)
        # We explicitly pass the seeds used in this run
        inference.run_ensemble_inference(
            seeds=RUN_SEEDS, output_path=config.SUBMISSION_FILE_PATH
        )
    else:
        print(
            f"Validation metric {final_metric:.6f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
