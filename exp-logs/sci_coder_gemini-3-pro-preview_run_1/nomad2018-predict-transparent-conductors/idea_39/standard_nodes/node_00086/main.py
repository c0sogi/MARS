import torch
import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import Engine
from library.data import get_train_val_loaders
from library.model import SCC_WDS_Net
from library.utils import set_seed


def calculate_column_wise_rmsle(preds, targets):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    Since inputs are already log(1+x) transformed, this is just the RMSE of the inputs.
    """
    # preds and targets are (N, 2) arrays in log space
    mse_per_col = np.mean((preds - targets) ** 2, axis=0)
    rmse_per_col = np.sqrt(mse_per_col)
    return np.mean(rmse_per_col)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Initialize Engine
    engine = Engine(device=device)

    # 3. Train Model
    # Using 50 epochs to ensure quick execution while allowing some convergence.
    # The dataset is small, so this should be very fast.
    print("Starting Training...")
    engine.run_training(epochs=50)

    # 4. Validation Assessment
    print("\nPerforming Validation Assessment...")
    # Reload validation loader (fast due to caching)
    _, val_loader = get_train_val_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Load best model
    model = SCC_WDS_Net().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print("Error: Model checkpoint not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic_feats, global_feats, batch_idx)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_global_feats.append(global_feats.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Calculate Metric
    # Inputs are already log-transformed, so RMSE on these IS the RMSLE on original scale
    metric = calculate_column_wise_rmsle(val_preds, val_targets)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample (in log space)
    sample_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Feature names based on geometry.py implementation
    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "frac_Al",
        "frac_Ga",
        "frac_In",
        "total_atoms",
    ]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(val_global_feats, columns=feature_names)
    analysis_df["error"] = sample_errors

    # Calculate correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Correlation between Input Features and Prediction Error:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.05479004207787702
    if metric < THRESHOLD:
        print(
            f"\nValidation metric ({metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        engine.predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
