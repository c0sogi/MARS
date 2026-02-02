import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import setup_logger, calculate_log_mae, Standardizer, set_seed
from library.dataset import FlattenedMoleculeDataset, collate_batch
from library.model import MoleculeModel


def main():
    # ==========================================
    # 0. Configuration Overrides for Fast Baseline
    # ==========================================
    # Force DEBUG mode to ensure the script completes within the time limit
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 8000  # Small subset for speed
    Config.MAX_EPOCHS = 3  # Minimal epochs for baseline
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 2

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Setup Logger and Seed
    logger = setup_logger(log_file=os.path.join(Config.WORK_DIR, "run.log"))
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # ==========================================
    # 1. Data Loading
    # ==========================================
    logger.info("Initializing Datasets...")
    # This will trigger preprocessing with DEBUG=True
    train_dataset = FlattenedMoleculeDataset(split="train", load_cached=True)
    val_dataset = FlattenedMoleculeDataset(split="val", load_cached=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # ==========================================
    # 2. Standardization
    # ==========================================
    logger.info("Setting up Standardizer...")
    standardizer = Standardizer()

    # Fit standardizer on training data
    # Cite solution_lesson_node_00017: Use canonical integer indices
    train_targets = train_dataset.data["target_values"]
    train_types_idx = train_dataset.data["target_types"]

    df_stats = pd.DataFrame(
        {"scalar_coupling_constant": train_targets, "type": train_types_idx}
    )
    standardizer.fit(df_stats)

    # Compute Aux stats for normalization
    train_shielding = train_dataset.data["aux_shielding"]
    train_charges = train_dataset.data["aux_charges"]

    shield_mean = torch.tensor(
        np.mean(train_shielding), dtype=torch.float32, device=device
    )
    shield_std = torch.tensor(
        np.std(train_shielding) + 1e-6, dtype=torch.float32, device=device
    )
    charge_mean = torch.tensor(
        np.mean(train_charges), dtype=torch.float32, device=device
    )
    charge_std = torch.tensor(
        np.std(train_charges) + 1e-6, dtype=torch.float32, device=device
    )

    # ==========================================
    # 3. Model & Optimizer
    # ==========================================
    logger.info("Initializing Model...")
    model = MoleculeModel().to(device)
    # Compile model for speedup on A100
    try:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")
    except Exception as e:
        logger.warning(f"Could not compile model: {e}")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.L1Loss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    logger.info("Starting Training...")
    best_metric = float("inf")

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            preds = model(batch)

            # Calculate Losses
            # 1. Primary Task
            z_targets = standardizer.transform(
                batch["target_values"], batch["target_types"]
            )
            loss_coupling = loss_fn(preds["scalar_coupling"].squeeze(), z_targets)

            # 2. Aux Tasks
            z_shield = (batch["aux_shielding"] - shield_mean) / shield_std
            loss_shield = loss_fn(preds["shielding"], z_shield)

            z_charge = (batch["aux_charges"] - charge_mean) / charge_std
            loss_charge = loss_fn(preds["charges"].squeeze(), z_charge)

            # Total Loss
            total_loss = loss_coupling + Config.AUX_LOSS_WEIGHT * (
                loss_shield + loss_charge
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()

        # Validation Step
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_types_list = []

        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                preds = model(batch)

                # Inverse transform predictions
                p_raw = standardizer.inverse_transform(
                    preds["scalar_coupling"].squeeze(), batch["target_types"]
                )

                val_preds_list.append(p_raw.cpu().numpy())
                val_targets_list.append(batch["target_values"].cpu().numpy())
                val_types_list.append(batch["target_types"].cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        val_types = np.concatenate(val_types_list)

        metric = calculate_log_mae(val_targets, val_preds, val_types)
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val LogMAE: {metric:.6f}"
        )

        if metric < best_metric:
            best_metric = metric
            torch.save(
                model.state_dict(), os.path.join(Config.WORK_DIR, "best_model.pth")
            )

    # ==========================================
    # 5. Final Validation & Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")
    model.load_state_dict(torch.load(os.path.join(Config.WORK_DIR, "best_model.pth")))
    model.eval()

    all_preds = []
    all_targets = []
    all_dists = []
    all_types = []

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            p_raw = standardizer.inverse_transform(
                preds["scalar_coupling"].squeeze(), batch["target_types"]
            )

            all_preds.append(p_raw.cpu().numpy())
            all_targets.append(batch["target_values"].cpu().numpy())
            all_types.append(batch["target_types"].cpu().numpy())

            # Extract distances for the target pairs
            # batch['target_indices'] points to edges. batch['edge_dist'] has distances.
            dists = batch["edge_dist"][batch["target_indices"]]
            all_dists.append(dists.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_dists = np.concatenate(all_dists)
    all_types = np.concatenate(all_types)

    # Calculate Final Metric
    final_metric = calculate_log_mae(all_targets, all_preds, all_types)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    abs_errors = np.abs(all_preds - all_targets)

    # Correlation with Distance
    if len(abs_errors) > 1:
        corr_dist, _ = pearsonr(abs_errors, all_dists)
        print(f"Correlation between Error and Distance: {corr_dist:.4f}")

        # Correlation with Target Magnitude
        corr_mag, _ = pearsonr(abs_errors, np.abs(all_targets))
        print(f"Correlation between Error and Target Magnitude: {corr_mag:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = -1.2761284112930298

    if final_metric < THRESHOLD:
        logger.info("Metric below threshold. Generating submission...")

        test_dataset = FlattenedMoleculeDataset(split="test", load_cached=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_batch,
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                preds = model(batch)
                p_raw = standardizer.inverse_transform(
                    preds["scalar_coupling"].squeeze(), batch["target_types"]
                )
                test_preds.append(p_raw.cpu().numpy())

        if len(test_preds) > 0:
            test_preds = np.concatenate(test_preds)
        else:
            test_preds = np.array([])

        # Reconstruct IDs
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Identify which molecules were processed (important if DEBUG mode subsampled test set)
        processed_mols_path = os.path.join(
            Config.CACHE_PATHS["test"], "processed_mols.npy"
        )
        if os.path.exists(processed_mols_path):
            relevant_mols = np.load(processed_mols_path, allow_pickle=True)
        else:
            relevant_mols = df_test_meta["molecule_name"].unique()

        # Align predictions with IDs based on dataset iteration order
        grouped = df_test_meta[
            df_test_meta["molecule_name"].isin(relevant_mols)
        ].groupby("molecule_name")

        id_list = []
        for mol in relevant_mols:
            if mol in grouped.groups:
                group = grouped.get_group(mol)
                id_list.append(group["id"].values)

        all_ids = np.concatenate(id_list)

        if len(all_ids) == len(test_preds):
            sub_df = pd.DataFrame(
                {"id": all_ids, "scalar_coupling_constant": test_preds}
            )
            sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
            sub_df.to_csv(sub_path, index=False)
            logger.info(f"Submission saved to {sub_path}")
        else:
            logger.error(
                f"Shape mismatch: IDs {len(all_ids)} vs Preds {len(test_preds)}"
            )
    else:
        logger.info(
            f"Metric {final_metric:.6f} not below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
