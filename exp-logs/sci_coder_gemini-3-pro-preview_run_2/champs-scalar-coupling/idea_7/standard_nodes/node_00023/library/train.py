import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.data import MolecularGraphDataset, collate_graphs, COUPLING_TYPES
from library.utils import TargetScaler, set_seed
from library.model import HGANet


def run_training(
    debug: bool = Config.DEBUG,
    max_epochs: int = Config.MAX_EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = Config.NUM_WORKERS,
):
    """
    Executes the training pipeline for the HGA-Net model.

    Args:
        debug (bool): If True, runs on a small subset of data.
        max_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate.
        device (str): Device to run training on ('cpu' or 'cuda').
        num_workers (int): Number of worker processes for data loading.
    """
    # Update Config with runtime arguments
    Config.DEBUG = debug
    Config.MAX_EPOCHS = max_epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate
    Config.DEVICE = torch.device(device)
    Config.NUM_WORKERS = num_workers

    # Set reproducible seed
    set_seed(Config.SEED)

    print(f"Starting run with DEBUG={Config.DEBUG}, Device={Config.DEVICE}")

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    print("Initializing Data Loaders...")
    # Note: MolecularGraphDataset reads Config.DEBUG internally
    train_dataset = MolecularGraphDataset(
        Config.TRAIN_CSV, "train", load_cached_data=True
    )
    val_dataset = MolecularGraphDataset(Config.VAL_CSV, "val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
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
    # 2. Target Scaling
    # ------------------------------------------------------------------
    print("Fitting Target Scaler...")
    scaler = TargetScaler()

    # Load training metadata for fitting scaler
    df_train = pd.read_csv(Config.TRAIN_CSV)
    if Config.DEBUG:
        # Filter df_train to match the debug subset used in dataset
        unique_mols = df_train["molecule_name"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        df_train = df_train[df_train["molecule_name"].isin(unique_mols)]

    scaler.fit(df_train)

    # Pre-compute stats tensors for efficient GPU normalization
    # Map stats to the sorted order of coupling types defined in library.data
    type_means = {}
    type_stds = {}
    for t, stats in scaler.stats.items():
        type_means[t] = stats["mean"]
        type_stds[t] = stats["std"]

    sorted_means = [type_means.get(t, 0.0) for t in COUPLING_TYPES]
    sorted_stds = [type_stds.get(t, 1.0) for t in COUPLING_TYPES]

    device_obj = Config.DEVICE
    tensor_means = torch.tensor(sorted_means, device=device_obj, dtype=torch.float32)
    tensor_stds = torch.tensor(sorted_stds, device=device_obj, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 3. Model Setup
    # ------------------------------------------------------------------
    print("Initializing Model...")
    model = HGANet(Config).to(device_obj)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    total_steps = Config.MAX_EPOCHS * steps_per_epoch

    # Use a safe percentage for warmup if epochs are very low (e.g. debug)
    # Cite debug_lesson_12
    if Config.MAX_EPOCHS > Config.WARMUP_EPOCHS:
        pct_start = Config.WARMUP_EPOCHS / Config.MAX_EPOCHS
    else:
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

    criterion = nn.L1Loss()

    # ------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------
    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device_obj)

            optimizer.zero_grad()

            # Forward pass
            preds = model(batch)

            # Normalize Targets on the fly
            # batch['coupling_type'] contains integer indices corresponding to COUPLING_TYPES
            batch_means_t = tensor_means[batch["coupling_type"]]
            batch_stds_t = tensor_stds[batch["coupling_type"]]

            targets = batch["coupling_value"]
            targets_norm = (targets - batch_means_t) / batch_stds_t

            loss = criterion(preds, targets_norm)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch["batch_size"]

        avg_train_loss = train_loss / len(train_dataset)

        # ------------------------------------------------------------------
        # 5. Validation
        # ------------------------------------------------------------------
        model.eval()
        all_preds_raw = []
        all_targets_raw = []
        all_types = []

        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device_obj)

                preds_norm = model(batch)

                # Denormalize predictions for metric calculation
                batch_means_t = tensor_means[batch["coupling_type"]]
                batch_stds_t = tensor_stds[batch["coupling_type"]]
                preds_raw = preds_norm * batch_stds_t + batch_means_t

                all_preds_raw.append(preds_raw.cpu().numpy())
                all_targets_raw.append(batch["coupling_value"].cpu().numpy())
                all_types.append(batch["coupling_type"].cpu().numpy())

        all_preds_raw = np.concatenate(all_preds_raw)
        all_targets_raw = np.concatenate(all_targets_raw)
        all_types = np.concatenate(all_types)

        # Calculate Log MAE per type and average
        unique_types = np.unique(all_types)
        log_maes = []
        for t_idx in unique_types:
            mask = all_types == t_idx
            mae = np.mean(np.abs(all_preds_raw[mask] - all_targets_raw[mask]))
            log_maes.append(np.log(mae + 1e-9))

        val_score = np.mean(log_maes)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.MAX_EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.6f} | Val LMAE: {val_score:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! ({val_score:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # ------------------------------------------------------------------
    # 6. Inference on Test Set
    # ------------------------------------------------------------------
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device_obj))
    model.eval()

    print("Processing Test Data...")
    test_dataset = MolecularGraphDataset(Config.TEST_CSV, "test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device_obj)

            preds_norm = model(batch)

            # Denormalize
            batch_means_t = tensor_means[batch["coupling_type"]]
            batch_stds_t = tensor_stds[batch["coupling_type"]]
            preds_raw = preds_norm * batch_stds_t + batch_means_t

            all_ids.append(batch["coupling_id"].cpu().numpy())
            all_preds.append(preds_raw.cpu().numpy())

    if len(all_ids) > 0:
        all_ids = np.concatenate(all_ids)
        all_preds = np.concatenate(all_preds)
    else:
        all_ids = np.array([], dtype=np.int64)
        all_preds = np.array([], dtype=np.float32)

    # Save Submission
    df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
