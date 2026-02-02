import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.data import MolecularGraphDataset, collate_graphs, COUPLING_TYPES
from library.model import HGANet
from library.utils import TargetScaler, set_seed


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Enforce fast baseline constraints
    Config.DEBUG = (
        False  # We need full validation set, so we turn off DEBUG global flag
    )
    Config.MAX_EPOCHS = 3
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 4

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    print("Loading datasets...")

    # Load full datasets (utilizing cache if available)
    train_dataset = MolecularGraphDataset(
        Config.TRAIN_CSV, "train", load_cached_data=True
    )
    val_dataset = MolecularGraphDataset(Config.VAL_CSV, "val", load_cached_data=True)

    # Create a subset for training to ensure speed (Fast Baseline)
    # Train on 15,000 molecules (~25% of data) to get a decent model quickly.
    num_train_mols = len(train_dataset)
    subset_size = min(15000, num_train_mols)
    indices = torch.randperm(num_train_mols)[:subset_size].tolist()

    train_subset = Subset(train_dataset, indices)

    print(f"Training on subset of {len(train_subset)} molecules.")
    print(f"Validating on full set of {len(val_dataset)} molecules.")

    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 3. Target Scaler
    # ------------------------------------------------------------------
    print("Fitting TargetScaler...")
    scaler = TargetScaler()
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    scaler.fit(df_train_meta)

    # Prepare normalization tensors
    type_means = {}
    type_stds = {}
    for t, stats in scaler.stats.items():
        type_means[t] = stats["mean"]
        type_stds[t] = stats["std"]

    # Create tensors mapped to COUPLING_TYPES indices (0-7)
    sorted_means = [type_means.get(t, 0.0) for t in COUPLING_TYPES]
    sorted_stds = [type_stds.get(t, 1.0) for t in COUPLING_TYPES]

    tensor_means = torch.tensor(sorted_means, device=device, dtype=torch.float32)
    tensor_stds = torch.tensor(sorted_stds, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. Model Initialization
    # ------------------------------------------------------------------
    print("Initializing HGANet...")
    model = HGANet(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.L1Loss()

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = Config.MAX_EPOCHS * steps_per_epoch
    pct_start = 0.1

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            # Forward
            preds = model(batch)

            # Normalize targets
            batch_means = tensor_means[batch["coupling_type"]]
            batch_stds = tensor_stds[batch["coupling_type"]]
            targets = batch["coupling_value"]
            targets_norm = (targets - batch_means) / batch_stds

            loss = criterion(preds, targets_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch["batch_size"]

        avg_train_loss = train_loss / len(train_subset)
        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.1f}s"
        )

    # ------------------------------------------------------------------
    # 6. Validation & Metric Calculation
    # ------------------------------------------------------------------
    print("Running Validation...")
    model.eval()

    all_errors = []
    all_dists = []

    # Accumulators for metric
    type_abs_errors = {t_idx: [] for t_idx in range(len(COUPLING_TYPES))}

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds_norm = model(batch)

            # Denormalize
            batch_means = tensor_means[batch["coupling_type"]]
            batch_stds = tensor_stds[batch["coupling_type"]]
            preds = preds_norm * batch_stds + batch_means

            targets = batch["coupling_value"]
            types = batch["coupling_type"].cpu().numpy()

            # Calculate errors
            abs_err = torch.abs(preds - targets).cpu().numpy()

            # Store for metric
            for i, t_idx in enumerate(types):
                type_abs_errors[t_idx].append(abs_err[i])

            # Calculate distances for failure analysis
            # coupling_atom_index is (C, 2)
            c_idx = batch["coupling_atom_index"]
            pos = batch["pos"]

            p0 = pos[c_idx[:, 0]]
            p1 = pos[c_idx[:, 1]]
            dists = torch.norm(p0 - p1, dim=1).cpu().numpy()

            all_errors.append(abs_err)
            all_dists.append(dists)

    # Compute Final Metric
    log_mae_list = []
    for t_idx in sorted(type_abs_errors.keys()):
        errors = np.array(type_abs_errors[t_idx])
        if len(errors) > 0:
            mae = np.mean(errors)
            log_mae = np.log(mae + 1e-9)
            log_mae_list.append(log_mae)

    final_metric = np.mean(log_mae_list)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # ------------------------------------------------------------------
    # 7. Failure Analysis
    # ------------------------------------------------------------------
    print("\nFailure Analysis:")
    all_errors = np.concatenate(all_errors)
    all_dists = np.concatenate(all_dists)

    # Correlation between Error and Distance
    corr_dist, _ = scipy.stats.pearsonr(all_errors, all_dists)
    print(
        f"Correlation between Absolute Error and Inter-atomic Distance: {corr_dist:.4f}"
    )

    # ------------------------------------------------------------------
    # 8. Conditional Submission
    # ------------------------------------------------------------------
    THRESHOLD = -1.407172441

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = MolecularGraphDataset(
            Config.TEST_CSV, "test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=Config.NUM_WORKERS,
        )

        all_ids = []
        all_preds_test = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                preds_norm = model(batch)

                # Denormalize
                batch_means = tensor_means[batch["coupling_type"]]
                batch_stds = tensor_stds[batch["coupling_type"]]
                preds = preds_norm * batch_stds + batch_means

                all_ids.append(batch["coupling_id"].cpu().numpy())
                all_preds_test.append(preds.cpu().numpy())

        if len(all_ids) > 0:
            all_ids = np.concatenate(all_ids)
            all_preds_test = np.concatenate(all_preds_test)

            df_sub = pd.DataFrame(
                {"id": all_ids, "scalar_coupling_constant": all_preds_test}
            )
            df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
