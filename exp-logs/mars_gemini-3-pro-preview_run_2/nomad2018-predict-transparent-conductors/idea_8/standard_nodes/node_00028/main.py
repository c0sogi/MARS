import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.train import Trainer
from library.data import CrystalGraphDataset, collate_batch
from library.utils import set_seed


def run():
    # 1. Setup
    set_seed(Config.SEED)

    # Initialize Trainer
    trainer = Trainer(Config)

    # 2. Train
    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.train(epochs=Config.EPOCHS)

    # 3. Validation & Metric Calculation
    print("Starting validation assessment...")

    # Load validation dataset
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        load_cached_data=True,
        debug=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Load best model state
    if os.path.exists(trainer.checkpoint_path):
        trainer.model.load_state_dict(
            torch.load(trainer.checkpoint_path, map_location=trainer.device)
        )
    else:
        print("Warning: No checkpoint found, using current model state.")

    trainer.model.eval()

    # Load target scaler
    if os.path.exists(trainer.target_scaler_path):
        trainer.target_scaler.load(trainer.target_scaler_path)

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(trainer.device)

            # Forward pass
            preds_norm = trainer.model(batch)

            # Inverse transform
            preds_orig = trainer.target_scaler.inverse_transform(preds_norm)

            all_preds.append(preds_orig.cpu())
            all_targets.append(batch.y.cpu())
            all_ids.append(batch.id.cpu())

    # Concatenate
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    ids = torch.cat(all_ids, dim=0).numpy().flatten()

    # Calculate Metric: Column-wise RMSLE
    # Ensure non-negative
    y_pred_clamped = torch.clamp(y_pred, min=0.0)
    y_true_clamped = torch.clamp(y_true, min=0.0)

    log_pred = torch.log1p(y_pred_clamped)
    log_true = torch.log1p(y_true_clamped)

    squared_log_error = (log_pred - log_true) ** 2
    mse_log_per_col = torch.mean(squared_log_error, dim=0)
    rmsle_per_col = torch.sqrt(mse_log_per_col)

    # Final metric is mean of column RMSLEs
    final_metric = torch.mean(rmsle_per_col).item()

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming failure analysis...")

    # Calculate error per sample (Mean Absolute Error per sample averaged over targets)
    abs_errors = torch.abs(y_pred - y_true).numpy()

    # Load validation metadata to get features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    results_df = pd.DataFrame(
        {
            "id": ids,
            "err_formation": abs_errors[:, 0],
            "err_bandgap": abs_errors[:, 1],
            "mean_abs_err": np.mean(abs_errors, axis=1),
        }
    )

    # Merge with features
    analysis_df = pd.merge(results_df, val_df, on="id", how="inner")

    # Select numerical features for correlation
    feature_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
    ]

    # Filter only columns that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in analysis_df.columns]

    print("Correlation between Mean Absolute Error and Features:")
    correlations = (
        analysis_df[feature_cols]
        .corrwith(analysis_df["mean_abs_err"])
        .sort_values(key=abs, ascending=False)
    )
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.05085437756413089

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation metric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
