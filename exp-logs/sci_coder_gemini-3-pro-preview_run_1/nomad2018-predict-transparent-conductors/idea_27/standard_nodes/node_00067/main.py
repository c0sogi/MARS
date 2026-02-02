import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.engine import Trainer
from library.data import get_dataloaders


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for a fast baseline execution
    # 50 epochs should be sufficient for convergence on this dataset size
    # while staying well within the time limit.
    Config.EPOCHS = 50

    # 2. Train Model
    trainer = Trainer()
    trainer.fit(epochs=Config.EPOCHS)

    # 3. Validation Assessment
    print("\nRunning Validation Assessment...")

    # Load validation dataloader (utilizing cached data)
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    trainer.model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atomic_feats = batch["atomic_features"].to(trainer.device)
            global_feats = batch["global_features"].to(trainer.device)
            batch_indices = batch["batch_indices"].to(trainer.device)
            targets = batch["targets"].to(trainer.device)
            ids = batch["id"]

            # Forward pass
            outputs = trainer.model(atomic_feats, global_feats, batch_indices)

            # Collect results
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate Metric
    # The model works in log1p space.
    # RMSLE(y_true, y_pred) = RMSE(log1p(y_true), log1p(y_pred))
    # Since all_targets and all_preds are already log1p transformed, we just calculate RMSE.

    mse_formation = mean_squared_error(all_targets[:, 0], all_preds[:, 0])
    mse_bandgap = mean_squared_error(all_targets[:, 1], all_preds[:, 1])

    rmsle_formation = np.sqrt(mse_formation)
    rmsle_bandgap = np.sqrt(mse_bandgap)

    # Final metric is the column-wise mean
    final_metric = (rmsle_formation + rmsle_bandgap) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude per sample (mean absolute error in log space)
    # This gives us a single scalar representing "how bad" the prediction was for each sample
    sample_errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Load validation metadata
    val_metadata_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if os.path.exists(val_metadata_path):
        val_df = pd.read_csv(val_metadata_path)

        # Create error dataframe
        error_df = pd.DataFrame({"id": all_ids, "error": sample_errors})

        # Merge with metadata
        analysis_df = val_df.merge(error_df, on="id")

        # Define features to correlate
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

        correlations = {}
        for col in feature_cols:
            if col in analysis_df.columns:
                # Calculate correlation
                corr = analysis_df[col].corr(analysis_df["error"])
                correlations[col] = corr

        print("Correlation between Error Magnitude and Input Features:")
        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feat, corr in sorted_corr:
            print(f"  {feat:<30}: {corr:.4f}")
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # 5. Submission
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
