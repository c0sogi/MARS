import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
import library.config as lib_config
from library.data import process_data, get_scalers, CrystalDataset, collate_sparse
from library.model import REMSWDSModel
from library.train import train_one_epoch, validate, predict

# -----------------------------------------------------------------------------
# CONFIGURATION OVERRIDES FOR FAST BASELINE
# -----------------------------------------------------------------------------
# Limit epochs for speed while ensuring convergence on this small dataset
lib_config.EPOCHS = 50
lib_config.BATCH_SIZE = 32
lib_config.LEARNING_RATE = 1e-3
lib_config.PATIENCE = 10

# Set seeds for reproducibility
SEED = lib_config.SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def run_failure_analysis(model, val_loader, device, scaler_global):
    """
    Performs failure analysis on the validation set.
    Correlates prediction error with global features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            af = batch["atomic_features"].to(device)
            idx = batch["batch_index"].to(device)
            gf = batch["global_features"].to(device)
            y = batch["targets"].to(device)

            outputs = model(af, idx, gf)

            all_targets.append(y.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_global_feats.append(gf.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_global_feats = np.vstack(all_global_feats)

    # Inverse transform global features to get physical values for correlation
    if scaler_global:
        physical_global_feats = scaler_global.inverse_transform(all_global_feats)
    else:
        physical_global_feats = all_global_feats

    # Calculate error per sample (Mean Squared Error per sample in log space)
    # shape: (N_samples, 2) -> mean over columns -> (N_samples,)
    sample_errors = np.mean((all_preds - all_targets) ** 2, axis=1)

    # Feature names based on extract_global_features in library/features.py
    # 3 lengths, 3 angles, 1 vol, 1 density, 3 stoichiometry, 1 total_atoms, 3 aspect ratios
    feature_names = [
        "lat_a",
        "lat_b",
        "lat_c",
        "alpha",
        "beta",
        "gamma",
        "volume",
        "density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "total_atoms",
        "ratio_ab",
        "ratio_bc",
        "ratio_ca",
    ]

    # Compute correlations
    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    correlations = []
    for i, name in enumerate(feature_names):
        if i < physical_global_feats.shape[1]:
            feat_values = physical_global_feats[:, i]
            # Handle potential constant features (std=0)
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, sample_errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"{name:<20} | {corr:.4f}")

    print("-" * 50)


def main():
    # -------------------------------------------------------------------------
    # 1. SETUP
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. DATA PREPARATION
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    train_df = pd.read_csv(lib_config.TRAIN_CSV)
    val_df = pd.read_csv(lib_config.VAL_CSV)
    test_df = pd.read_csv(lib_config.TEST_CSV)

    print("Processing data (loading from cache if available)...")
    # Load/Compute features
    train_af, train_gf, train_y, train_ids = process_data(
        train_df, lib_config.TRAIN_CACHE_PATH
    )
    val_af, val_gf, val_y, val_ids = process_data(val_df, lib_config.VAL_CACHE_PATH)
    test_af, test_gf, test_y, test_ids = process_data(
        test_df, lib_config.TEST_CACHE_PATH
    )

    # Fit scalers on training data
    scaler_atomic, scaler_global = get_scalers(train_af, train_gf)

    # Create Datasets
    train_dataset = CrystalDataset(
        train_af,
        train_gf,
        train_y,
        train_ids,
        scaler_atomic,
        scaler_global,
        mode="train",
    )
    val_dataset = CrystalDataset(
        val_af, val_gf, val_y, val_ids, scaler_atomic, scaler_global, mode="val"
    )
    # Test dataset (mode='test' skips log transform on targets)
    test_dataset = CrystalDataset(
        test_af, test_gf, test_y, test_ids, scaler_atomic, scaler_global, mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=lib_config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=lib_config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=lib_config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse,
    )

    # -------------------------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    print("Initializing REMS-WDS Model...")
    model = REMSWDSModel(
        atom_features_dim=lib_config.ATOM_FEATURES_DIM,
        global_features_dim=lib_config.GLOBAL_FEATURES_DIM,
        hidden_dim=lib_config.HIDDEN_DIM,
        atomic_layers=lib_config.ATOMIC_LAYERS,
        global_layers=lib_config.GLOBAL_LAYERS,
        fusion_layers=lib_config.FUSION_LAYERS,
        dropout=lib_config.DROPOUT,
        use_bn=lib_config.USE_BATCH_NORM,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lib_config.LEARNING_RATE,
        weight_decay=lib_config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # -------------------------------------------------------------------------
    # 4. TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"Starting training for {lib_config.EPOCHS} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(lib_config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), lib_config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= lib_config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. VALIDATION & METRIC
    # -------------------------------------------------------------------------
    # Load best model
    model.load_state_dict(torch.load(lib_config.MODEL_SAVE_PATH))
    model.eval()

    # Compute final validation metric
    # The criterion is MSE on log-transformed targets.
    # RMSLE = sqrt(MSE)
    final_val_loss = validate(model, val_loader, criterion, device)
    final_metric = np.sqrt(final_val_loss)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    run_failure_analysis(model, val_loader, device, scaler_global)

    # -------------------------------------------------------------------------
    # 7. SUBMISSION
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05366557091474533

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Predict on test set
        # predict() function handles the loop and inverse transform (expm1)
        raw_preds, ids = predict(model, test_loader, device)

        # Format dataframe
        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": raw_preds[:, 0],
                "bandgap_energy_ev": raw_preds[:, 1],
            }
        )

        # Save
        os.makedirs(os.path.dirname(lib_config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(lib_config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {lib_config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
