import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, StandardScaler
from library.data import get_train_val_datasets, get_test_dataset
from library.model import GCCGCNN
from library.engine import train_one_epoch, evaluate


def calculate_rmsle(preds, targets):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    RMSLE = mean( sqrt( mean( (log1p(pred) - log1p(target))^2 ) ) )
    """
    # Clip predictions to be non-negative to avoid log domain errors
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    log_preds = np.log1p(preds)
    log_targets = np.log1p(targets)

    squared_diff = (log_preds - log_targets) ** 2
    mse = np.mean(squared_diff, axis=0)
    rmse = np.sqrt(mse)

    return np.mean(rmse)


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset, val_dataset = get_train_val_datasets(load_cached=True)
    test_dataset = get_test_dataset(load_cached=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Target Scaling
    print("Fitting scaler...")
    all_train_y = []
    for data in train_loader:
        all_train_y.append(data.y)
    all_train_y = torch.cat(all_train_y, dim=0)

    scaler = StandardScaler(device=device)
    scaler.fit(all_train_y)

    # 4. Model Initialization
    print("Initializing model...")
    model = GCCGCNN(config=Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    epochs = Config.NUM_EPOCHS

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validation
        val_loss, _, _ = evaluate(model, val_loader, criterion, scaler, device)

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            break

    # 6. Final Evaluation
    # Load best model
    print("Loading best model...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get predictions on validation set
    print("Performing final validation...")
    _, _, val_predictions = evaluate(model, val_loader, criterion, scaler, device)
    val_ids = val_predictions["ids"]
    val_preds = val_predictions["preds"]

    # Reconstruct targets from loader (order is preserved as shuffle=False)
    val_targets_list = []
    for data in val_loader:
        val_targets_list.append(data.y.numpy())
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Compute Metric
    final_metric = calculate_rmsle(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude per sample (MAE across targets)
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Load metadata
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)
    # Align metadata with predictions using ID
    val_meta_df = val_meta_df.set_index("id")
    val_meta_df = val_meta_df.reindex(val_ids)

    # Select numeric features for correlation
    numeric_cols = val_meta_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude targets from features
    for t in Config.TARGET_COLS:
        if t in numeric_cols:
            numeric_cols.remove(t)

    # Compute correlations
    correlations = {}
    for col in numeric_cols:
        # Handle potential NaNs in features
        feat_values = val_meta_df[col].fillna(0).values
        if len(np.unique(feat_values)) > 1:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top correlations between error magnitude and features:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 8. Submission
    threshold = 0.05085437756413089
    if final_metric < threshold:
        print("Metric meets threshold. Generating submission...")
        _, _, test_predictions = evaluate(model, test_loader, criterion, scaler, device)
        test_ids = test_predictions["ids"]
        test_preds = test_predictions["preds"]

        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} is not lower than threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
