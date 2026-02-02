import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data import get_datasets, CollateFn
from library.model import SIRDS_SP
from library.train import Trainer
from library.predict import generate_submission
from library.utils import set_seed


def calculate_column_wise_rmsle(preds, targets):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    Since the model predicts log1p(target), this is equivalent to RMSE on the model outputs.

    Args:
        preds (torch.Tensor): Model predictions (log space).
        targets (torch.Tensor): Ground truth (log space).

    Returns:
        float: The mean of the RMSLEs for each target column.
    """
    # MSE per column
    mse_col = torch.mean((preds - targets) ** 2, dim=0)
    # RMSE per column (which is RMSLE in original space)
    rmsle_col = torch.sqrt(mse_col)
    # Mean of column-wise RMSLEs
    metric = torch.mean(rmsle_col).item()
    return metric


def perform_failure_analysis(model, val_loader, val_dataset, device):
    """
    Correlates prediction errors with input features to identify failure modes.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()
    all_errors = []
    all_indices = []

    # 1. Compute errors
    with torch.no_grad():
        for batch in val_loader:
            atom_features = batch["atom_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            spacegroups = batch["spacegroups"].to(device)
            targets = batch["targets"].to(device)

            preds = model(atom_features, batch_indices, global_features, spacegroups)

            # Error in log space (magnitude of residual)
            # We take the mean absolute error across the two targets for each sample
            batch_errors = torch.mean(torch.abs(preds - targets), dim=1).cpu().numpy()
            all_errors.extend(batch_errors)

            # Keep track of indices to map back to raw features
            # The loader might shuffle, but we are in validation mode (shuffle=False usually)
            # However, to be safe, we rely on the order of the loader matching the dataset iteration

    all_errors = np.array(all_errors)

    # 2. Get Raw Global Features for correlation
    # val_dataset.data['global_features'] contains the raw physics features
    # We need to ensure we align them correctly. The val_loader iterates sequentially.
    raw_features = val_dataset.data["global_features"]

    # Feature names based on process_data implementation
    feature_names = [
        "lattice_vec_1",
        "lattice_vec_2",
        "lattice_vec_3",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "percent_al",
        "percent_ga",
        "percent_in",
        "total_atoms",
        "volume",
        "density",
    ]

    # Create DataFrame
    df_analysis = pd.DataFrame(raw_features, columns=feature_names)
    df_analysis["error"] = all_errors

    # 3. Compute Correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.to_string())
    print("-" * 40)

    # Identify worst failures
    worst_idx = np.argmax(all_errors)
    print(f"Worst prediction error: {all_errors[worst_idx]:.6f}")
    print(f"Worst sample features:\n{df_analysis.iloc[worst_idx]}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Load cached data to speed up startup
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    collate_fn = CollateFn()

    # Use appropriate batch size and workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SI-RDS-SP Model...")
    model = SIRDS_SP()

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader)

    # 5. Final Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No best model checkpoint found. Using current weights.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            atom_features = batch["atom_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            spacegroups = batch["spacegroups"].to(device)
            targets = batch["targets"].to(device)

            preds = model(atom_features, batch_indices, global_features, spacegroups)

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Metric
    final_metric = calculate_column_wise_rmsle(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, val_dataset, device)

    # 7. Submission
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=device,
        )
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
