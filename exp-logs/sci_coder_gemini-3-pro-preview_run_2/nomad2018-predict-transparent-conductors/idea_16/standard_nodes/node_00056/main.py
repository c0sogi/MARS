import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, compute_rmsle
from library.data import CrystalGraphDataset, collate_graphs
from library.model import IR_CGCNN
from library.train import train_one_epoch, validate


def run_failure_analysis(val_df, preds, targets, target_cols):
    """
    Analyzes correlations between error magnitude and input features.
    """
    print("\nFailure Analysis:")
    print("-" * 20)

    # Calculate per-sample error magnitude (L1 norm of error vector for simplicity)
    errors = np.abs(preds - targets)

    # Add errors to dataframe
    for i, target_name in enumerate(target_cols):
        val_df[f"error_{target_name}"] = errors[:, i]

    # Select numerical features for correlation
    feature_cols = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Calculate correlations
    for i, target_name in enumerate(target_cols):
        print(f"Correlations for error in {target_name}:")
        error_col = f"error_{target_name}"
        correlations = []
        for feat in feature_cols:
            if feat in val_df.columns:
                # Handle potential NaNs just in case
                valid_mask = ~val_df[feat].isna() & ~val_df[error_col].isna()
                if valid_mask.sum() > 1:
                    corr, _ = pearsonr(
                        val_df.loc[valid_mask, feat], val_df.loc[valid_mask, error_col]
                    )
                    correlations.append((feat, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, corr in correlations[:5]:
            print(f"  {feat:<30}: {corr:.4f}")
        print()


def main():
    # 1. Setup
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 30

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Datasets...")
    # Load cached data if available to speed up
    train_dataset = CrystalGraphDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    val_dataset = CrystalGraphDataset(
        Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = IR_CGCNN(Config).to(device)

    # 4. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 5. Training Loop
    print("Starting Training...")
    best_val_rmsle = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        # Note: validate function in library/train.py calculates RMSLE on unscaled data
        val_loss, val_rmsle = validate(
            model, val_loader, criterion, device, train_dataset.scaler
        )

        # Scheduler step
        scheduler.step(val_rmsle)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmsle": val_rmsle,
                },
                Config.BEST_MODEL_PATH,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Validation Assessment
    print("\nPerforming Final Validation Assessment...")

    # Load best model
    checkpoint = load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    model.eval()

    # We need predictions and targets for failure analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for atom_fea, edge_index, edge_fea, batch_index, targets, _ in val_loader:
            atom_fea = atom_fea.to(device)
            edge_index = edge_index.to(device)
            edge_fea = edge_fea.to(device)
            batch_index = batch_index.to(device)

            preds = model(atom_fea, edge_index, edge_fea, batch_index)

            preds_np = preds.cpu().numpy()
            targets_np = targets.numpy()

            # Inverse transform
            preds_unscaled = train_dataset.scaler.inverse_transform(preds_np)
            targets_unscaled = train_dataset.scaler.inverse_transform(targets_np)

            all_preds.append(preds_unscaled)
            all_targets.append(targets_unscaled)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Ensure non-negative
    all_preds = np.maximum(all_preds, 0)

    # Compute Final Metric
    final_metric = compute_rmsle(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # val_dataset.metadata is the correct subset corresponding to the validation loader
    run_failure_analysis(
        val_dataset.metadata.copy(), all_preds, all_targets, Config.TARGET_COLS
    )

    # 8. Submission Generation
    threshold = 0.05085437756413089
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_dataset = CrystalGraphDataset(
            Config.TEST_METADATA_PATH, mode="test", load_cached_data=True
        )
        # Manually set scaler for test dataset from training
        test_dataset.scaler = train_dataset.scaler

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
        )

        ids_list = []
        test_preds_list = []

        with torch.no_grad():
            for atom_fea, edge_index, edge_fea, batch_index, _, ids in test_loader:
                atom_fea = atom_fea.to(device)
                edge_index = edge_index.to(device)
                edge_fea = edge_fea.to(device)
                batch_index = batch_index.to(device)

                preds = model(atom_fea, edge_index, edge_fea, batch_index)

                preds_np = preds.cpu().numpy()
                preds_unscaled = test_dataset.scaler.inverse_transform(preds_np)

                ids_list.extend(ids.numpy())
                test_preds_list.append(preds_unscaled)

        all_test_preds = np.concatenate(test_preds_list, axis=0)

        submission_df = pd.DataFrame(
            {
                "id": ids_list,
                "formation_energy_ev_natom": all_test_preds[:, 0],
                "bandgap_energy_ev": all_test_preds[:, 1],
            }
        )

        # Clip negative values
        submission_df["formation_energy_ev_natom"] = submission_df[
            "formation_energy_ev_natom"
        ].clip(lower=0.0)
        submission_df["bandgap_energy_ev"] = submission_df["bandgap_energy_ev"].clip(
            lower=0.0
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission generation skipped.")


if __name__ == "__main__":
    main()
