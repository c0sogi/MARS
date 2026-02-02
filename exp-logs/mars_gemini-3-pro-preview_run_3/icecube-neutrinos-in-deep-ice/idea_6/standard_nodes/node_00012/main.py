import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the current directory is in the python path to load the library modules
sys.path.append(os.getcwd())

from library.config import Config
from library import data, model, train, inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    # We use a subset of data for training to ensure the run completes quickly (<< 2 hours).
    # We use 200,000 samples and 3 epochs.
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200000
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 512

    # Setup directories and reproducibility
    Config.setup_directories()
    train.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing Training Data (Subset)...")
    # Train on subset (DEBUG=True)
    train_dataset = data.IceCubeDataset(mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # Use a small validation set for monitoring during training
    val_dataset_small = data.IceCubeDataset(mode="val")
    val_loader_small = DataLoader(
        val_dataset_small,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    net = model.CFDGN().to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_val_mae = float("inf")

    print(f"Starting Training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train one epoch
        t_loss, t_mae = train.train_one_epoch(net, train_loader, optimizer, device)

        # Validate on small set
        v_loss, v_mae = train.validate(net, val_loader_small, device)

        # Step scheduler
        scheduler.step()

        # Save best model
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save(net.state_dict(), Config.MODEL_PATH)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {duration:.1f}s | "
            f"Train MAE: {t_mae:.4f} | Val MAE (Subset): {v_mae:.4f}"
        )

    # -------------------------------------------------------------------------
    # 4. Full Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPreparing for Full Validation...")

    # Switch to Full Data Mode
    Config.DEBUG = False

    # Re-instantiate validation dataset to load the full metadata
    val_dataset_full = data.IceCubeDataset(mode="val")
    val_loader_full = DataLoader(
        val_dataset_full,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load best model weights
    net.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    net.eval()

    print("Running Full Validation and Failure Analysis...")
    total_mae = 0.0
    total_samples = 0

    # Storage for analysis
    all_errors = []
    all_n_pulses = []
    all_mean_charge = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader_full):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            batch_size = x.size(0)

            # Forward pass
            pred = net(x, mask)

            # Calculate angular error (radians)
            cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            angles = torch.acos(cos_sim)

            # Accumulate metric
            total_mae += angles.sum().item()
            total_samples += batch_size

            # Accumulate data for analysis
            all_errors.append(angles.cpu().numpy())

            # Extract features: N Pulses
            n_p = mask.sum(dim=1).cpu().numpy()
            all_n_pulses.append(n_p)

            # Extract features: Mean Charge
            # x[:, :, 4] is log10(charge + 1)
            charges = x[:, :, 4]
            sum_q = (charges * mask.float()).sum(dim=1).cpu().numpy()
            mean_q = np.divide(sum_q, n_p, out=np.zeros_like(sum_q), where=n_p != 0)
            all_mean_charge.append(mean_q)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} batches...")

    # Compute Final Metric
    final_metric = total_mae / total_samples
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    print("\nFailure Analysis:")
    errors_flat = np.concatenate(all_errors)
    n_pulses_flat = np.concatenate(all_n_pulses)
    mean_charge_flat = np.concatenate(all_mean_charge)

    corr_pulses = np.corrcoef(errors_flat, n_pulses_flat)[0, 1]
    corr_charge = np.corrcoef(errors_flat, mean_charge_flat)[0, 1]

    print(f"Correlation (Error vs N_Pulses): {corr_pulses:.6f}")
    print(f"Correlation (Error vs Mean_Charge): {corr_charge:.6f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 1.5013689469017657

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Proceeding to submission...")
        # inference.run_inference will load the full test set since Config.DEBUG is False
        inference.run_inference(
            model_path=Config.MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            device=Config.DEVICE,
        )
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
