import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import LI_CGCNN_ELR
from library.train import train_one_epoch, evaluate, save_checkpoint, load_checkpoint
from library.utils import set_seed, StandardScaler


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    """
    # Ensure non-negative
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_error = (log_true - log_pred) ** 2
    mse = np.mean(squared_error, axis=0)
    rmsle_per_column = np.sqrt(mse)

    return np.mean(rmsle_per_column)


def perform_failure_analysis(
    model, val_loader, val_metadata_path, device, target_scaler
):
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds_scaled = model(batch)

            # Inverse transform
            preds = target_scaler.inverse_transform(preds_scaled.cpu().numpy())
            targets = target_scaler.inverse_transform(batch.y.cpu().numpy())

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSLE
    rmsle = calculate_rmsle(all_targets, all_preds)
    print(f"Final Validation Metric: {rmsle}")

    # Calculate Mean Absolute Error per sample (averaged over targets for analysis)
    # We analyze error magnitude
    abs_errors = np.abs(all_preds - all_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)  # Shape (N_val,)

    # Load metadata to correlate with features
    val_df = pd.read_csv(val_metadata_path)

    # Ensure alignment (DataLoader preserves order if shuffle=False)
    if len(val_df) != len(mean_abs_error):
        print(
            "Warning: Metadata length does not match prediction length. Skipping correlation analysis."
        )
        return rmsle

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

    print("\nCorrelation between Mean Absolute Error and Features:")
    correlations = {}
    for col in feature_cols:
        if col in val_df.columns:
            corr = np.corrcoef(val_df[col].values, mean_abs_error)[0, 1]
            correlations[col] = corr

    # Sort and print
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs:
        print(f"  {feat:<30}: {corr:.4f}")

    return rmsle


def generate_test_submission(model, test_loader, device, target_scaler):
    print("\nGenerating submission for test set...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_scaled = model(batch)
            preds = target_scaler.inverse_transform(preds_scaled.cpu().numpy())
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    # Load test metadata for IDs
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    ids = test_meta["id"].values

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Override Config for Fast Baseline
    # We use fewer epochs to ensure it finishes quickly within the limit
    Config.NUM_EPOCHS = 20

    # 2. Data Loading
    # load_cached_data=True will try to use existing .npz files in ./working/idea_18/cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Load Target Scaler for inverse transform later
    target_scaler = StandardScaler()
    if os.path.exists(Config.TARGET_SCALER_PATH):
        target_scaler.load(Config.TARGET_SCALER_PATH)
    else:
        raise FileNotFoundError(
            "Target scaler not found. Ensure get_dataloaders generated it."
        )

    # 3. Model Initialization
    model = LI_CGCNN_ELR(Config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # 5. Load Best Model
    print(f"\nLoading best model from {Config.BEST_MODEL_PATH}")
    epoch, loss = load_checkpoint(
        model, None, Config.BEST_MODEL_PATH, device=Config.DEVICE
    )
    print(f"Restored model from epoch {epoch} with Val Loss (MSE scaled): {loss:.6f}")

    # 6. Validation Assessment & Failure Analysis
    # This prints "Final Validation Metric: <value>"
    val_rmsle = perform_failure_analysis(
        model, val_loader, Config.VAL_METADATA_PATH, device, target_scaler
    )

    # 7. Conditional Submission
    # Threshold from instructions: 0.049412816762924194
    THRESHOLD = 0.049412816762924194

    if val_rmsle < THRESHOLD:
        print(f"\nValidation metric {val_rmsle} < {THRESHOLD}. Generating submission.")
        generate_test_submission(model, test_loader, device, target_scaler)
    else:
        print(f"\nValidation metric {val_rmsle} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
