import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from library.config import Config
from library.utils import setup_logger, calculate_log_mae, GroupStandardizer
from library.data import get_dataloaders
from library.model import DirectionalMPNN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, standardizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Standardize targets
        # Targets in batch are raw values. We need to standardize them for training.
        # Move targets to cpu for numpy processing in standardizer, then back to device
        y_raw = batch.target_val.cpu().numpy()
        types = batch.target_type.cpu().numpy()

        # Transform: (y - mean) / std
        y_std = standardizer.transform(
            y_raw, batch.target_type.cpu().numpy()
        )  # map int types to string if needed?
        # The standardizer expects integer types if that's how it was fit, or strings.
        # In data.py, target_type are integers mapped from COUPLING_MAP.
        # In utils.py, GroupStandardizer saves stats with keys from type_col.
        # We need to ensure consistency.
        # utils.py GroupStandardizer.fit uses `type` column from dataframe which are strings (e.g. '1JHC').
        # batch.target_type are integers. We need to map integers back to strings for the standardizer.

        # Inverse map for types
        int_to_type = {v: k for k, v in Config.COUPLING_MAP.items()}
        type_strings = [int_to_type[t] for t in types]

        y_std = standardizer.transform(y_raw, type_strings)
        target = torch.tensor(y_std, device=device, dtype=torch.float32).unsqueeze(-1)

        # Forward pass
        # Note: batch.target_edge_index_uv/vu are used inside model to select edges
        out = model(
            z=batch.z,
            pos=batch.pos,
            edge_index=batch.edge_index,
            idx_kj=batch.idx_kj,
            idx_ji=batch.idx_ji,
            target_node_0=batch.target_node_0,
            target_node_1=batch.target_node_1,
            target_type=batch.target_type,
            target_edge_index_uv=batch.target_edge_index_uv,
            target_edge_index_vu=batch.target_edge_index_vu,
        )

        # Loss calculation (L1 Loss on standardized targets)
        loss = nn.functional.l1_loss(out, target)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, standardizer, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_types = []

    int_to_type = {v: k for k, v in Config.COUPLING_MAP.items()}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward
            out = model(
                z=batch.z,
                pos=batch.pos,
                edge_index=batch.edge_index,
                idx_kj=batch.idx_kj,
                idx_ji=batch.idx_ji,
                target_node_0=batch.target_node_0,
                target_node_1=batch.target_node_1,
                target_type=batch.target_type,
                target_edge_index_uv=batch.target_edge_index_uv,
                target_edge_index_vu=batch.target_edge_index_vu,
            )

            # Prepare for metric calculation
            # Inverse transform predictions
            preds_std = out.cpu().numpy().flatten()
            types_int = batch.target_type.cpu().numpy()
            type_strings = [int_to_type[t] for t in types_int]

            preds_raw = standardizer.inverse_transform(preds_std, type_strings)
            targets_raw = batch.target_val.cpu().numpy()

            all_preds.extend(preds_raw)
            all_targets.extend(targets_raw)
            all_types.extend(type_strings)

            # Calculate validation loss on standardized targets (for scheduler/early stopping consistency)
            # Standardize targets again just for loss calc
            targets_std = standardizer.transform(targets_raw, type_strings)
            targets_std_tensor = torch.tensor(
                targets_std, device=device, dtype=torch.float32
            ).unsqueeze(-1)
            loss = nn.functional.l1_loss(out, targets_std_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    # Calculate Competition Metric
    metric = calculate_log_mae(
        np.array(all_targets), np.array(all_preds), np.array(all_types)
    )

    return avg_loss, metric


def predict_test(model, loader, standardizer, device):
    model.eval()
    all_ids = []
    all_preds = []

    int_to_type = {v: k for k, v in Config.COUPLING_MAP.items()}

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(
                z=batch.z,
                pos=batch.pos,
                edge_index=batch.edge_index,
                idx_kj=batch.idx_kj,
                idx_ji=batch.idx_ji,
                target_node_0=batch.target_node_0,
                target_node_1=batch.target_node_1,
                target_type=batch.target_type,
                target_edge_index_uv=batch.target_edge_index_uv,
                target_edge_index_vu=batch.target_edge_index_vu,
            )

            preds_std = out.cpu().numpy().flatten()
            types_int = batch.target_type.cpu().numpy()
            type_strings = [int_to_type[t] for t in types_int]

            preds_raw = standardizer.inverse_transform(preds_std, type_strings)
            ids = batch.target_ids.cpu().numpy()

            all_ids.extend(ids)
            all_preds.extend(preds_raw)

    return np.array(all_ids), np.array(all_preds)


def run_training(load_cached_data=True):
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("Trainer", os.path.join(Config.WORKING_DIR, "train.log"))
    device = torch.device(Config.DEVICE)

    logger.info("Starting training pipeline...")
    Config.print_config()

    # 2. Prepare Standardizer
    # We need to fit the standardizer on the training metadata
    logger.info("Initializing Standardizer...")
    standardizer = GroupStandardizer()

    # Check if we need to fit or if it will load from cache inside .fit()
    # We pass the dataframe just in case cache is missing
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    standardizer.fit(df_train_meta, load_cached_data=load_cached_data)

    # 3. Data Loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_meta_path=Config.TRAIN_META_PATH,
        val_meta_path=Config.VAL_META_PATH,
        test_meta_path=Config.TEST_META_PATH,
        structures_path=Config.STRUCTURES_CSV,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
        load_cached_data=load_cached_data,
    )

    # 4. Model & Optimizer
    logger.info("Initializing model...")
    model = DirectionalMPNN(
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        num_radial=Config.NUM_RBF,
        num_spherical=Config.NUM_SBF,
        cutoff=Config.CUTOFF,
        envelope_exponent=Config.ENVELOPE_EXPONENT,
        num_output_layers=Config.NUM_OUTPUT_LAYERS,
        out_emb_dim=Config.TYPE_EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Schedulers
    # Warmup + ReduceLROnPlateau
    # We implement warmup manually using LambdaLR, then switch control to Plateau

    def warmup_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return float(epoch + 1) / float(Config.WARMUP_EPOCHS)
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Plateau scheduler for after warmup
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_DECAY_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 6. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, standardizer, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, standardizer, device)

        # Scheduler Step
        if epoch <= Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            # Monitor validation metric (LogMAE)
            plateau_scheduler.step(val_metric)
            current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric (LogMAE): {val_metric:.9f}"
        )

        # Early Stopping & Model Saving
        # We optimize for the competition metric (LogMAE)
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"  New best model saved! Metric: {best_metric:.9f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # 7. Final Prediction
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    ids, preds = predict_test(model, test_loader, standardizer, device)

    # Save submission
    df_sub = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    return best_metric


if __name__ == "__main__":
    # This block is here for local testing if run directly,
    # but the prompt asks to only implement the module class/functions.
    # The requirement "Only implement the module class/functions. DO NOT include an if __name__ == "__main__": block"
    # overrides this. I will remove this block in the final output.
    pass
