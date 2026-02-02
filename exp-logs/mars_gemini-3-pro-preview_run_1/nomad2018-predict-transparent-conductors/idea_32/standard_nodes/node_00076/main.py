import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data import MaterialDataset, collate_batch
from library.model import ChemicallyWeightedDeepSets
from library.train import Trainer, train_model, generate_submission


def main():
    # 1. Configure for Fast Baseline
    # We override the number of epochs to ensure the run completes quickly within the time limit
    # while still allowing convergence on this dataset size.
    print("Configuring for fast baseline run...")
    Config.NUM_EPOCHS = 50
    Config.BATCH_SIZE = 64  # Keep batch size reasonable

    # 2. Train Model
    # This function handles dataset initialization, scaling, and the training loop.
    # It saves the best model to Config.MODEL_CHECKPOINT.
    print("Starting training...")
    train_model(load_cached_data=True)

    # 3. Validation and Failure Analysis
    print("Performing validation and failure analysis...")

    # Load Validation Data
    # We use the validation set defined in metadata
    val_dataset = MaterialDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
    )

    # Load the best model trained in step 2
    device = Config.DEVICE
    model = ChemicallyWeightedDeepSets()
    checkpoint_path = Config.MODEL_CHECKPOINT

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds_log = []
    all_targets_log = []
    all_ids = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            atomic_mask = batch["atomic_mask"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(atomic_features, atomic_mask, global_features)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_ids.append(ids.numpy())

    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    # Calculate Metric (Column-wise RMSLE)
    # The model predicts log(1+y), and targets are log(1+y).
    # RMSE on these log-values is equivalent to RMSLE on the original values.
    mse_col1 = mean_squared_error(all_targets_log[:, 0], all_preds_log[:, 0])
    mse_col2 = mean_squared_error(all_targets_log[:, 1], all_preds_log[:, 1])

    rmsle_1 = np.sqrt(mse_col1)
    rmsle_2 = np.sqrt(mse_col2)
    final_metric = (rmsle_1 + rmsle_2) / 2

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Mean Absolute Error on original scale for analysis
    preds_original = np.expm1(all_preds_log)
    targets_original = np.expm1(all_targets_log)

    # Average MAE across the two targets for each sample
    errors = np.abs(preds_original - targets_original)
    mean_error_per_sample = np.mean(errors, axis=1)

    # Load original validation metadata to get feature values for correlation
    val_df = pd.read_csv(Config.VAL_METADATA)
    # Ensure alignment by ID
    val_df.set_index("id", inplace=True)
    # Reindex to match the order of predictions
    val_df = val_df.loc[all_ids]

    # Add error to dataframe
    val_df["model_error"] = mean_error_per_sample

    # Calculate correlations between numerical features and model error
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    # Drop targets from correlation check to focus on input features
    cols_to_check = [
        c
        for c in numeric_cols
        if c not in ["formation_energy_ev_natom", "bandgap_energy_ev", "model_error"]
    ]

    correlations = (
        val_df[cols_to_check]
        .corrwith(val_df["model_error"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop Correlations with Model Error:")
    print(correlations.head(10))

    # 4. Submission
    # Generate submission if metric is good enough
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
