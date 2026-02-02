import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.data import CrystalDataset, collate_sparse_batch
from library.model import GBAMSDSModel
from library.train import train_epoch, validate

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure execution within time limits while allowing convergence
Config.EPOCHS = 50
Config.BATCH_SIZE = 64  # Increase batch size for speed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_global_feature_names():
    """Helper to generate names for global features for analysis."""
    names = []
    # Lattice (6)
    names.extend(["a", "b", "c", "alpha", "beta", "gamma"])
    # Vol, Dens (2)
    names.extend(["volume", "density"])
    # Stoich (4)
    names.extend([f"stoich_{el}" for el in Config.ATOM_LIST])
    # N_atoms (1)
    names.append("n_atoms")
    # Aspect (3)
    names.extend(["aspect_ab", "aspect_bc", "aspect_ca"])
    # Physics (3)
    names.extend(["mean_mass", "mean_radius", "mean_eneg"])
    # Bond Stats (10)
    for p in Config.BOND_PAIRS:
        names.append(f"bond_{p[0]}_{p[1]}")
    return names


def main():
    print("Starting GBA-MS-DS Pipeline...")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("Loading Datasets...")

    # Train Dataset (Fits scalers)
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_file=Config.TRAIN_CACHE_FILE,
        fit_scalers=True,
        transform_targets=True,
        load_cached_data=True,
    )

    # Validation Dataset (Uses training scalers)
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_file=Config.VAL_CACHE_FILE,
        scalers=(train_dataset.a_scaler, train_dataset.g_scaler),
        fit_scalers=False,
        transform_targets=True,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse_batch,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = GBAMSDSModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=False,
    )

    # MSE Loss on log-transformed targets minimizes RMSLE on original scale
    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    print(f"Training for {Config.EPOCHS} epochs...")
    best_val_metric = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        # Note: validate() in library returns (avg_loss, rmsle_form, rmsle_band)
        # The metric is column-wise RMSLE.
        # Since targets are log1p, RMSE on these targets IS RMSLE on original.
        val_loss, rmsle_form, rmsle_band = validate(
            model, val_loader, criterion, device
        )

        # Metric defined in task: Column-wise root mean squared logarithmic error
        # This implies mean(RMSLE_col1, RMSLE_col2)
        current_metric = (rmsle_form + rmsle_band) / 2.0

        scheduler.step(val_loss)

        # Checkpointing
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training Complete.")

    # -------------------------------------------------------------------------
    # 4. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Calculate Final Metric on full validation set
    _, final_rmsle_form, final_rmsle_band = validate(
        model, val_loader, criterion, device
    )
    final_metric = (final_rmsle_form + final_rmsle_band) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_errors = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "atomic_features": batch["atomic_features"].to(device),
                "global_features": batch["global_features"].to(device),
                "batch_indices": batch["batch_indices"].to(device),
            }
            targets = batch["targets"].to(device)

            outputs = model(inputs)

            # Error on log scale (which corresponds to RMSLE contribution)
            # shape: (B, 2)
            errors = torch.abs(outputs - targets)
            val_errors.append(errors.cpu().numpy())

            # Collect global features for correlation (inverse transform scaling if possible,
            # but correlation works on scaled data too)
            val_global_feats.append(batch["global_features"].cpu().numpy())

    val_errors = np.concatenate(val_errors, axis=0)  # (N, 2)
    val_global_feats = np.concatenate(val_global_feats, axis=0)  # (N, 29)

    # Mean error across the two targets for analysis
    mean_errors = np.mean(val_errors, axis=1)

    feature_names = get_global_feature_names()

    # Calculate correlations
    correlations = []
    for i in range(val_global_feats.shape[1]):
        feat_col = val_global_feats[:, i]
        # Handle constant features (std=0)
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, mean_errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.04819517582654953

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = CrystalDataset(
            metadata_path=Config.TEST_METADATA_PATH,
            cache_file=Config.TEST_CACHE_FILE,
            scalers=(train_dataset.a_scaler, train_dataset.g_scaler),
            fit_scalers=False,
            transform_targets=False,
            load_cached_data=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_sparse_batch,
            num_workers=2,
        )

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    "atomic_features": batch["atomic_features"].to(device),
                    "global_features": batch["global_features"].to(device),
                    "batch_indices": batch["batch_indices"].to(device),
                }
                ids = batch["ids"]

                # Forward
                outputs_log = model(inputs)

                # Inverse Transform: exp(x) - 1
                outputs_original = torch.expm1(outputs_log)

                all_ids.append(ids.cpu().numpy())
                all_preds.append(outputs_original.cpu().numpy())

        all_ids = np.concatenate(all_ids)
        all_preds = np.concatenate(all_preds)

        submission_df = pd.DataFrame(
            {
                "id": all_ids,
                "formation_energy_ev_natom": all_preds[:, 0],
                "bandgap_energy_ev": all_preds[:, 1],
            }
        )

        # Sort by ID
        submission_df.sort_values("id", inplace=True)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
