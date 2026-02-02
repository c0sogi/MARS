import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import library modules
from library.config import Config
from library.utils import TargetScaler, LogMAE
from library.dataset import CouplingDataset, collate_graphs
from library.model import HGANet


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for efficient A100 usage and time management
    Config.BATCH_SIZE = 192  # Increase batch size for A100
    Config.EPOCHS = 3  # Limit epochs for fast baseline
    Config.NUM_WORKERS = 12  # Use all vCPUs
    Config.DEBUG = False  # Use full dataset

    # Set seeds
    Config.set_seed(Config.SEED)

    print(f"Running Experiment: {Config.EXPERIMENT_NAME}")
    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("\n--- Data Preparation ---")

    # Load metadata
    train_meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize and fit TargetScaler
    print("Fitting TargetScaler...")
    scaler = TargetScaler()
    scaler.fit(train_meta_df, load_cache=True)

    # Pre-compute scaler stats as tensors for fast GPU usage
    # types are 0-7 based on Geometry.COUPLING_TYPE_MAP
    # We create a tensor of shape (8,) for means and stds
    type_map = {
        v: k for k, v in scaler.means.items()
    }  # Ensure we map correctly if keys are strings
    # But scaler.means keys are strings (e.g., '1JHC').
    # Geometry.COUPLING_TYPE_MAP maps '1JHC' -> 0.
    # We need arrays indexed by the integer type.

    means_arr = np.zeros(8, dtype=np.float32)
    stds_arr = np.ones(8, dtype=np.float32)

    # Invert the geometry map to get string from int
    from library.geometry import Geometry

    int_to_type = {v: k for k, v in Geometry.COUPLING_TYPE_MAP.items()}

    for i in range(8):
        t_str = int_to_type.get(i)
        if t_str in scaler.means:
            means_arr[i] = scaler.means[t_str]
            stds_arr[i] = scaler.stds[t_str]

    means_tensor = torch.tensor(means_arr, device=Config.DEVICE)
    stds_tensor = torch.tensor(stds_arr, device=Config.DEVICE)

    # Initialize Datasets
    print("Initializing Training Dataset...")
    train_dataset = CouplingDataset(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_PATH,
        load_cached_data=True,
        split="train",
    )

    print("Initializing Validation Dataset...")
    val_dataset = CouplingDataset(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_PATH,
        load_cached_data=True,
        split="val",
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n--- Model Initialization ---")
    model = HGANet().to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR for super-convergence
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE * 10,  # Allow higher peak
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    criterion = nn.L1Loss()
    scaler_amp = GradScaler()  # For Mixed Precision

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(Config.DEVICE)

            # Prepare targets
            targets = batch["coupling_targets"]
            types = batch["coupling_types"]

            # Normalize targets using pre-computed tensors
            # (x - mean) / std
            targets_scaled = (targets - means_tensor[types]) / (
                stds_tensor[types] + 1e-8
            )
            targets_scaled = targets_scaled.unsqueeze(1)  # (N, 1)

            optimizer.zero_grad()

            # Mixed Precision Forward & Backward
            with autocast():
                preds = model(batch)
                loss = criterion(preds, targets_scaled)

            scaler_amp.scale(loss).backward()

            # Gradient Clipping
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            epoch_loss += loss.item()

            if (i + 1) % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | Step {i+1}/{steps_per_epoch} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.1f}s"
        )

    # Save Model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 5. Validation & Metric
    # -------------------------------------------------------------------------
    print("\n--- Validation ---")
    model.eval()

    all_preds = []
    all_targets = []
    all_types = []

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(Config.DEVICE)

            # Forward
            with autocast():
                preds_scaled = model(batch)

            # Collect data for metric calculation
            # We need to inverse transform predictions
            preds_scaled_cpu = preds_scaled.detach().cpu().numpy().flatten()
            types_cpu = batch["coupling_types"].cpu().numpy()
            targets_cpu = batch["coupling_targets"].cpu().numpy()

            # Inverse transform: x * std + mean
            # We use the scaler class method for convenience on CPU arrays
            types_str = [int_to_type[t] for t in types_cpu]
            preds_orig = scaler.inverse_transform(preds_scaled_cpu, types_str)

            all_preds.append(preds_orig)
            all_targets.append(targets_cpu)
            all_types.append(types_cpu)

    # Concatenate
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_types = np.concatenate(all_types)

    # Map integer types back to strings for LogMAE
    y_types_str = [int_to_type[t] for t in y_types]

    # Compute Metric
    final_score, score_dict = LogMAE.score(y_true, y_pred, y_types_str)

    print(f"Final Validation Metric: {final_score:.9f}")
    print("Scores per type:", score_dict)

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Reconstruct validation dataframe to align features
    # We must replicate the sorting order of CouplingDataset
    grouped = val_meta_df.groupby("molecule_name")
    unique_mols = sorted(list(grouped.groups.keys()))  # Sorted keys

    val_ordered_list = []
    for mol in unique_mols:
        val_ordered_list.append(grouped.get_group(mol))

    val_aligned_df = pd.concat(val_ordered_list, ignore_index=True)

    # Add predictions and errors
    val_aligned_df["pred"] = y_pred
    val_aligned_df["abs_error"] = np.abs(
        val_aligned_df["scalar_coupling_constant"] - val_aligned_df["pred"]
    )

    # Calculate correlations
    # We need to compute distance again or load it if we had it.
    # Since we don't have distance in metadata easily without loading structures,
    # we will correlate with available metadata columns and error magnitude.

    # Just correlate with target magnitude and simple available features if any
    # We can also check error by type
    print("Mean Absolute Error by Type:")
    print(val_aligned_df.groupby("type")["abs_error"].mean())

    # Correlation with target magnitude
    corr_target = val_aligned_df["abs_error"].corr(
        val_aligned_df["scalar_coupling_constant"].abs()
    )
    print(f"Correlation between Error and Target Magnitude: {corr_target:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = -1.407172441

    if final_score < THRESHOLD:
        print(
            f"\nMetric {final_score:.9f} is better than {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_dataset = CouplingDataset(
            Config.TEST_METADATA_PATH,
            Config.CACHE_TEST_PATH,
            load_cached_data=True,
            split="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_graphs,
            pin_memory=True,
        )

        test_preds = []
        test_ids = []  # We need to reconstruct IDs or assume order

        # To ensure IDs match, we should align with test_metadata
        # CouplingDataset processes in sorted molecule order.
        # We will load test_metadata, sort by molecule, and extract IDs.
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
        grouped_test = test_meta_df.groupby("molecule_name")
        unique_mols_test = sorted(list(grouped_test.groups.keys()))

        test_ordered_list = []
        for mol in unique_mols_test:
            test_ordered_list.append(grouped_test.get_group(mol))

        test_aligned_df = pd.concat(test_ordered_list, ignore_index=True)
        submission_ids = test_aligned_df["id"].values

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(Config.DEVICE)

                with autocast():
                    preds_scaled = model(batch)

                preds_scaled_cpu = preds_scaled.detach().cpu().numpy().flatten()
                types_cpu = batch["coupling_types"].cpu().numpy()

                types_str = [int_to_type[t] for t in types_cpu]
                preds_orig = scaler.inverse_transform(preds_scaled_cpu, types_str)
                test_preds.append(preds_orig)

        final_preds = np.concatenate(test_preds)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id": submission_ids, "scalar_coupling_constant": final_preds}
        )

        # Sort by ID to match sample submission format (usually sorted by ID)
        submission_df = submission_df.sort_values("id")

        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")

    else:
        print(
            f"\nMetric {final_score:.9f} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
