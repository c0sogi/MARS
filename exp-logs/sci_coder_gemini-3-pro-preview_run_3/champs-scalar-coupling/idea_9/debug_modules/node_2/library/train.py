import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import setup_logger, calculate_log_mae, Standardizer, set_seed
from library.dataset import FlattenedMoleculeDataset, collate_batch
from library.model import MoleculeModel


def train_model():
    # ==========================================
    # 1. Setup
    # ==========================================
    logger = setup_logger(log_file=os.path.join(Config.WORK_DIR, "train.log"))
    set_seed(Config.SEED)
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading Datasets...")
    # Load cached data (preprocessing.py handles creation if cache misses)
    train_dataset = FlattenedMoleculeDataset(split="train", load_cached=True)
    val_dataset = FlattenedMoleculeDataset(split="val", load_cached=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=Config.PIN_MEMORY,
    )

    # ==========================================
    # 3. Standardization
    # ==========================================
    logger.info("Loading Target Standardizer...")
    standardizer = Standardizer()
    stats_path = Config.CACHE_PATHS["stats"]
    if os.path.exists(stats_path):
        standardizer.load(stats_path)
    else:
        logger.warning(
            f"Standardizer stats not found at {stats_path}. Training might be unstable."
        )

    # Compute Global Stats for Auxiliary Tasks (Shielding & Charges)
    # We access the raw numpy arrays directly from the dataset for efficiency
    logger.info("Computing Auxiliary Target Statistics...")
    train_shielding = train_dataset.data["aux_shielding"]
    train_charges = train_dataset.data["aux_charges"]

    # Compute mean/std and move to device for fast usage in loop
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
    # 4. Model Initialization
    # ==========================================
    logger.info("Initializing Model...")
    model = MoleculeModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=Config.MIN_LR
    )

    l1_loss_fn = nn.L1Loss()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_metric = float("inf")
    patience = Config.SCHEDULER_PATIENCE + 5
    patience_counter = 0

    logger.info(f"Starting training for {Config.MAX_EPOCHS} epochs...")

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            # Forward Pass
            preds = model(batch)

            # --- Loss Calculation ---

            # 1. Primary Task: Scalar Coupling
            # Transform targets to z-scores
            z_targets = standardizer.transform(
                batch["target_values"], batch["target_types"]
            )
            loss_coupling = l1_loss_fn(preds["scalar_coupling"].squeeze(), z_targets)

            # 2. Aux Task: Magnetic Shielding
            # Normalize target (N_nodes, 9)
            z_shield = (batch["aux_shielding"] - shield_mean) / shield_std
            loss_shield = l1_loss_fn(preds["shielding"], z_shield)

            # 3. Aux Task: Mulliken Charges
            # Normalize target (N_nodes,) -> unsqueeze or squeeze pred
            z_charge = (batch["aux_charges"] - charge_mean) / charge_std
            loss_charge = l1_loss_fn(preds["charges"].squeeze(), z_charge)

            # Composite Loss
            total_loss = loss_coupling + Config.AUX_LOSS_WEIGHT * (
                loss_shield + loss_charge
            )

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ==========================================
        # 6. Validation
        # ==========================================
        model.eval()
        val_preds = []
        val_targets = []
        val_types = []

        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                preds = model(batch)

                # Inverse Transform Predictions for Metric Calculation
                p_raw = standardizer.inverse_transform(
                    preds["scalar_coupling"].squeeze(), batch["target_types"]
                )

                val_preds.append(p_raw.cpu().numpy())
                val_targets.append(batch["target_values"].cpu().numpy())
                val_types.append(batch["target_types"].cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_types = np.concatenate(val_types)

        # Calculate LogMAE
        metric = calculate_log_mae(val_targets, val_preds, val_types)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val LogMAE: {metric:.6f} | Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(Config.WORK_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # ==========================================
    # 7. Prediction & Submission
    # ==========================================
    logger.info("Loading best model for submission...")
    model.load_state_dict(torch.load(os.path.join(Config.WORK_DIR, "best_model.pth")))
    model.eval()

    test_dataset = FlattenedMoleculeDataset(split="test", load_cached=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_batch,
    )

    all_preds = []
    logger.info("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)

            # Inverse transform
            p_raw = standardizer.inverse_transform(
                preds["scalar_coupling"].squeeze(), batch["target_types"]
            )
            all_preds.append(p_raw.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    # Reconstruct IDs
    # We must replicate the order of molecules used in preprocessing to align predictions with IDs
    logger.info("Reconstructing IDs...")
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load the exact list of processed molecules to ensure alignment
    processed_mols_path = os.path.join(Config.CACHE_PATHS["test"], "processed_mols.npy")
    if os.path.exists(processed_mols_path):
        relevant_mols = np.load(processed_mols_path, allow_pickle=True)
    else:
        # Fallback to metadata if cache not found (legacy/debug behavior)
        if Config.DEBUG:
            unique_mols = df_test_meta["molecule_name"].unique()
            if len(unique_mols) > Config.DEBUG_SAMPLE_SIZE:
                rng = np.random.RandomState(Config.SEED)
                selected_mols = rng.choice(
                    unique_mols, Config.DEBUG_SAMPLE_SIZE, replace=False
                )
                df_test_meta = df_test_meta[
                    df_test_meta["molecule_name"].isin(selected_mols)
                ].reset_index(drop=True)
        relevant_mols = df_test_meta["molecule_name"].unique()

    # Group by molecule to match dataset structure
    # FlattenedMoleculeDataset processes molecules in the order of `unique()` appearance
    # and then iterates targets within that molecule group.
    grouped = df_test_meta.groupby("molecule_name")

    id_list = []
    for mol in relevant_mols:
        group = grouped.get_group(mol)
        id_list.append(group["id"].values)

    all_ids = np.concatenate(id_list)

    # Sanity check
    if len(all_ids) != len(all_preds):
        logger.error(f"Shape Mismatch! IDs: {len(all_ids)}, Preds: {len(all_preds)}")
    else:
        sub_df = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")
