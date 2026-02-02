import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_log_error

# Import from library
from library.config import Config
from library.data import get_dataloaders
from library.model import VNCGCNN
from library.train import train_one_epoch, validate, set_seed

# Override Config for fast baseline execution
Config.NUM_EPOCHS = 60
Config.BATCH_SIZE = 48


def calculate_mcrmsle(y_true, y_pred):
    """
    Calculates Mean Column-wise Root Mean Squared Logarithmic Error.
    """
    # Clip predictions to be non-negative as log is undefined for negative numbers
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    rmsles = []
    for i in range(y_true.shape[1]):
        # MSLE calculation
        msle = mean_squared_log_error(y_true[:, i], y_pred[:, i])
        rmsles.append(np.sqrt(msle))

    return np.mean(rmsles)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # load_cached_data=True will try to use existing cache in working/idea_6
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = VNCGCNN(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    criterion = MSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # 6. Evaluation on Validation Set
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    val_ids = []
    val_preds_scaled = []
    val_targets_scaled = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)

            val_ids.extend(batch.id.cpu().numpy().flatten())
            val_preds_scaled.append(outputs.cpu().numpy())
            val_targets_scaled.append(batch.y.cpu().numpy())

    val_preds_scaled = np.concatenate(val_preds_scaled, axis=0)
    val_targets_scaled = np.concatenate(val_targets_scaled, axis=0)

    # Inverse transform
    if target_scaler:
        val_preds = target_scaler.inverse_transform(val_preds_scaled)
        val_targets = target_scaler.inverse_transform(val_targets_scaled)
    else:
        val_preds = val_preds_scaled
        val_targets = val_targets_scaled

    # Compute Metric
    final_metric = calculate_mcrmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Calculate Mean Absolute Error per sample
    sample_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Load metadata
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create error dataframe
    error_df = pd.DataFrame({"id": val_ids, "error": sample_errors})

    # Merge with metadata
    analysis_df = val_meta_df.merge(error_df, on="id")

    # Select numerical features for correlation
    feature_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "error"]
    feature_cols = [c for c in feature_cols if c not in exclude]

    print("Failure Analysis (Correlation with Error):")
    correlations = {}
    for col in feature_cols:
        if analysis_df[col].std() > 0:  # Avoid constant columns
            corr = analysis_df[col].corr(analysis_df["error"])
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Print sorted correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs:
        print(f"{feat}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.05085437756413089

    if final_metric < THRESHOLD:
        test_ids = []
        test_preds_scaled = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)

                test_ids.extend(batch.id.cpu().numpy().flatten())
                test_preds_scaled.append(outputs.cpu().numpy())

        test_preds_scaled = np.concatenate(test_preds_scaled, axis=0)

        if target_scaler:
            test_preds = target_scaler.inverse_transform(test_preds_scaled)
        else:
            test_preds = test_preds_scaled

        # Create submission dataframe
        sub_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID
        sub_df = sub_df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Validation metric too high, skipping submission.")


if __name__ == "__main__":
    main()
