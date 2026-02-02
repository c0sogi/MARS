import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import LRCGCNN
from library.utils import set_seed, StandardScaler
from library.train import train_one_epoch, evaluate, generate_submission


def calculate_rmsle(preds, targets):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    preds: (N, 2) array
    targets: (N, 2) array
    """
    # Ensure non-negative for log input (clipping at 0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    log_preds = np.log1p(preds)
    log_targets = np.log1p(targets)

    squared_diff = (log_preds - log_targets) ** 2
    # Mean over samples first, then sqrt, then mean over columns
    rmse_per_col = np.sqrt(np.mean(squared_diff, axis=0))
    return np.mean(rmse_per_col)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True utilizes preprocessed .npz files for speed
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = LRCGCNN(
        atom_fea_len=Config.ATOM_FEA_LEN,
        h_fea_len=Config.H_FEA_LEN,
        n_conv=Config.N_CONV,
        n_h=Config.N_H,
        n_rbf=Config.N_RBF,
        radius=Config.RADIUS,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=8, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # 5. Validation & Metric Calculation
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    val_ids = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform to get original scale (eV)
            preds_orig = scaler.inverse_transform(outputs)
            targets_orig = scaler.inverse_transform(batch.y)

            val_ids.extend(batch.material_id.cpu().numpy())
            val_preds.append(preds_orig.cpu().numpy())
            val_targets.append(targets_orig.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    metric = calculate_rmsle(val_preds, val_targets)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude (mean absolute error across the two targets)
    errors = np.abs(val_preds - val_targets).mean(axis=1)

    # Load metadata to get features
    val_metadata = pd.read_csv(Config.VAL_METADATA)

    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({"id": val_ids, "error": errors})

    # Merge with metadata on 'id'
    analysis_df = analysis_df.merge(val_metadata, on="id", how="left")

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
    print("Correlation between Error Magnitude and Features:")
    correlations = analysis_df[feature_cols].corrwith(analysis_df["error"])
    print(correlations.sort_values(key=abs, ascending=False))

    # 7. Conditional Submission
    threshold = 0.05085437756413089
    if metric < threshold:
        print(f"Metric {metric} < {threshold}. Generating submission...")
        generate_submission(model, test_loader, scaler, device, Config.SUBMISSION_FILE)
    else:
        print(f"Metric {metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
