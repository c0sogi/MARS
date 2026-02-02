import sys
import os
import time
import torch
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

# Import provided library modules
from library.config import Config
from library.dataset import IceCubeDataset
from library.model import DFCGN
from library.loss import CosineSimilarityLoss
from library.train import train_one_epoch, validate, set_seeds
from library.inference import predict_and_submit
from library.utils import load_sensor_geometry


def perform_failure_analysis(model, loader, device):
    """
    Analyzes model performance by correlating error with event features.
    """
    print("\n=== Performing Failure Analysis ===")
    model.eval()
    all_errors = []
    all_n_pulses = []
    all_mean_charge = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            # 1. Compute Angular Error per event
            pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1)

            # Target is (azimuth, zenith)
            azimuth = batch.y[:, 0]
            zenith = batch.y[:, 1]

            # Convert target angles to vector
            sin_zen = torch.sin(zenith)
            tx = torch.cos(azimuth) * sin_zen
            ty = torch.sin(azimuth) * sin_zen
            tz = torch.cos(zenith)
            target_vec = torch.stack([tx, ty, tz], dim=1).to(pred.dtype)

            # Cosine similarity & Angle
            cosine_sim = torch.sum(pred_norm * target_vec, dim=1)
            cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            angles = torch.acos(cosine_sim)

            all_errors.append(angles.cpu().numpy())

            # 2. Extract Features
            # Number of pulses: bincount of batch index
            n_pulses = torch.bincount(batch.batch)
            all_n_pulses.append(n_pulses.cpu().numpy())

            # Mean Charge (Feature index 7 is log10(charge))
            charges = batch.x[:, 7].unsqueeze(1)
            mean_q = global_mean_pool(charges, batch.batch)
            all_mean_charge.append(mean_q.flatten().cpu().numpy())

    # Concatenate results
    errors = np.concatenate(all_errors)
    n_pulses = np.concatenate(all_n_pulses)
    mean_charge = np.concatenate(all_mean_charge)

    # Compute Correlations
    corr_pulses = np.corrcoef(errors, n_pulses)[0, 1]
    corr_charge = np.corrcoef(errors, mean_charge)[0, 1]

    print(f"Correlation between Error and N_Pulses: {corr_pulses:.4f}")
    print(f"Correlation between Error and Mean_Charge: {corr_charge:.4f}")

    if corr_pulses < 0:
        print("-> Higher pulse count tends to reduce error.")
    else:
        print("-> Higher pulse count tends to increase error.")


def cache_test_batch(bid):
    """
    Helper function to trigger caching for a single test batch.
    Used for parallel processing.
    """
    try:
        # Initializing the dataset triggers _process_and_cache_batch
        _ = IceCubeDataset(mode="test", batch_ids=[bid])
    except Exception as e:
        return e
    return None


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 512
    Config.NUM_WORKERS = 12

    # Ensure submission path is correct
    submission_path = Path("./submission/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    Config.SUBMISSION_PATH = submission_path

    # Define data subsets for speed (Batches 1-5 for train, 200-201 for val)
    # This processes ~1M training events and ~400k validation events.
    TRAIN_BATCH_IDS = list(range(1, 6))
    VAL_BATCH_IDS = [200, 201]

    print("=== Starting Fast Baseline Run ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Training Batches: {TRAIN_BATCH_IDS}")
    print(f"Validation Batches: {VAL_BATCH_IDS}")

    set_seeds(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure geometry is cached once before parallel processing
    load_sensor_geometry()

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Training Data...")
    train_dataset = IceCubeDataset(mode="train", batch_ids=TRAIN_BATCH_IDS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Loading Validation Data...")
    val_dataset = IceCubeDataset(mode="val", batch_ids=VAL_BATCH_IDS)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    model = DFCGN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = CosineSimilarityLoss()

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n=== Training Start ===")
    best_val_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        start_t = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_t
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"\nTraining finished. Best Val MAE: {best_val_mae:.6f}")

    # ---------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # ---------------------------------------------------------
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on the validation subset
    _, final_metric = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    perform_failure_analysis(model, val_loader, device)

    # ---------------------------------------------------------
    # 6. Submission Logic
    # ---------------------------------------------------------
    THRESHOLD = 1.5013689469017657

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} < {THRESHOLD}. Proceeding to submission generation..."
        )

        # Parallelize Test Data Pre-processing
        # This is critical to finish within the time limit as single-threaded processing is too slow.
        print("Pre-processing test batches in parallel...")

        test_df = pd.read_parquet(Config.TEST_METADATA_PATH, columns=["batch_id"])
        test_batch_ids = sorted(test_df["batch_id"].unique())

        start_proc = time.time()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=Config.NUM_WORKERS
        ) as executor:
            list(executor.map(cache_test_batch, test_batch_ids))
        print(f"Test data caching complete in {time.time() - start_proc:.1f}s.")

        # Run Inference
        # We use a larger batch size for inference to maximize GPU throughput
        print("Running inference and generating submission...")
        try:
            predict_and_submit(batch_size=1024, num_workers=Config.NUM_WORKERS)
        except Exception as e:
            print(f"Error during submission generation: {e}")

    else:
        print(f"\nMetric {final_metric:.6f} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
