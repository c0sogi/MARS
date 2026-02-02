import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from library.config import CONFIG
from library.data import get_data_loaders
from library.model import PADSDS
from library.engine import Engine


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_column_wise_rmsle(preds_log, targets_log):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    Since inputs are already log(1+x), this is equivalent to RMSE on the inputs.
    """
    mse = np.mean((preds_log - targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse)
    return np.mean(rmsle_per_col)


def perform_failure_analysis(global_feats, preds_log, targets_log):
    """
    Correlates error magnitude with global features.
    """
    # Calculate absolute error in log space (which corresponds to ratio error in linear space)
    errors = np.abs(preds_log - targets_log)
    mean_error = np.mean(
        errors, axis=1
    )  # Average error across the two targets for correlation

    # Feature names based on library/features.py construction
    feature_names = [
        "lattice_vector_1",
        "lattice_vector_2",
        "lattice_vector_3",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "total_atoms",
        "pct_al",
        "pct_ga",
        "pct_in",
    ]

    print(
        "\nFailure Analysis (Correlation between Mean Absolute Log-Error and Global Features):"
    )
    print("-" * 80)
    print(f"{'Feature':<20} | {'Correlation':<12}")
    print("-" * 80)

    correlations = []
    for i in range(global_feats.shape[1]):
        if i < len(feature_names):
            feat_name = feature_names[i]
        else:
            feat_name = f"feature_{i}"

        feat_values = global_feats[:, i]
        # Handle constant features (std=0) to avoid NaN correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, mean_error)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"{name:<20} | {corr:.4f}")
    print("-" * 80)


def main():
    # 1. Setup
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load cached data if available to speed up
    train_loader, val_loader, test_loader, scaler_atomic, scaler_global = (
        get_data_loaders(
            input_dir="./input", batch_size=CONFIG["batch_size"], load_cached_data=True
        )
    )

    # Determine input dimensions from a sample batch
    sample_atomic, _, sample_global, _, _ = next(iter(train_loader))
    atomic_input_dim = sample_atomic.shape[1]
    global_input_dim = sample_global.shape[1]

    print(f"Atomic Input Dim: {atomic_input_dim}")
    print(f"Global Input Dim: {global_input_dim}")

    # 3. Model Initialization
    model = PADSDS(
        atomic_input_dim=atomic_input_dim,
        global_input_dim=global_input_dim,
        hidden_dim=CONFIG["hidden_dim_atomic"],
        latent_dim=CONFIG["latent_dim"],
    ).to(device)

    # 4. Training
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )

    # Loss function: MSE on log-transformed targets
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    engine = Engine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        patience=CONFIG["patience"],
    )

    engine.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    # 5. Validation & Metrics
    print("\nRunning Final Validation...")
    # Load best model
    model.load_state_dict(torch.load(engine.best_model_path, map_location=device))
    model.eval()

    val_preds_log = []
    val_targets_log = []
    val_globals = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            outputs = model(atomic_feats, batch_indices, global_feats)

            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(targets.cpu().numpy())
            val_globals.append(global_feats.cpu().numpy())

    val_preds_log = np.concatenate(val_preds_log, axis=0)
    val_targets_log = np.concatenate(val_targets_log, axis=0)
    val_globals = np.concatenate(val_globals, axis=0)

    # Calculate Metric
    # Targets are already log(1+x). Model predicts log(1+x).
    # RMSLE is RMSE of log values.
    final_metric = calculate_column_wise_rmsle(val_preds_log, val_targets_log)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_globals, val_preds_log, val_targets_log)

    # 7. Submission
    threshold = 0.05479004207787702
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        engine.generate_submission(
            test_loader, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
