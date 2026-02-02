import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Import from library
from library.config import Config
from library.data import get_dataloaders
from library.model import MCPDSModel
from library.train import Trainer, set_seed


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    set_seed(Config.SEED)

    # Override epochs for fast execution as per requirements
    Config.EPOCHS = 50

    print("Initializing MC-PDS Pipeline...")
    Config.print_config()

    # 2. Data Loading
    print("\n[Data Loading]")
    # Load cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("\n[Model Initialization]")
    model = MCPDSModel()
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters."
    )

    # 4. Training
    print("\n[Training]")
    trainer = Trainer(model)
    trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    )

    # 5. Validation Assessment
    print("\n[Validation Assessment]")
    model.eval()

    all_preds_log = []
    all_targets_log = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_features"].to(Config.DEVICE)
            global_feats = batch["global_features"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)
            targets = batch["targets"].to(Config.DEVICE)
            ids = batch["ids"]

            outputs = model(atomic_feats, global_feats, mask)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_ids.extend(ids.numpy())

    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)

    # Calculate Column-wise RMSLE
    # Since model predicts log1p(y) and targets are log1p(y),
    # RMSE on these values IS the RMSLE.

    rmse_col1 = np.sqrt(mean_squared_error(all_targets_log[:, 0], all_preds_log[:, 0]))
    rmse_col2 = np.sqrt(mean_squared_error(all_targets_log[:, 1], all_preds_log[:, 1]))

    final_metric = (rmse_col1 + rmse_col2) / 2.0

    print(f"RMSLE Formation Energy: {rmse_col1:.6f}")
    print(f"RMSLE Bandgap Energy: {rmse_col2:.6f}")
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n[Failure Analysis]")
    # Calculate mean absolute error per sample in log space (proxy for relative error)
    sample_errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Load validation metadata to correlate with features
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if os.path.exists(val_meta_path):
        val_df = pd.read_csv(val_meta_path)

        # Map errors to dataframe using IDs
        # Create a mapping dict
        error_map = {id_: err for id_, err in zip(all_ids, sample_errors)}

        # Add error column
        val_df["model_error"] = val_df["id"].map(error_map)

        # Select numerical columns for correlation
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove id and targets from correlation check
        cols_to_check = [
            c
            for c in numeric_cols
            if c
            not in [
                "id",
                "formation_energy_ev_natom",
                "bandgap_energy_ev",
                "model_error",
            ]
        ]

        if cols_to_check:
            correlations = (
                val_df[cols_to_check]
                .corrwith(val_df["model_error"])
                .abs()
                .sort_values(ascending=False)
            )
            print("Top 5 features correlated with model error:")
            print(correlations.head(5))
        else:
            print("No numerical features available for correlation analysis.")
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # 7. Submission
    threshold = 0.05479004207787702
    if final_metric < threshold:
        print(f"\n[Submission]")
        print(f"Metric {final_metric} < {threshold}. Generating submission...")
        trainer.generate_submission(test_loader, Config.SUBMISSION_PATH)
    else:
        print(f"\n[Submission]")
        print(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
