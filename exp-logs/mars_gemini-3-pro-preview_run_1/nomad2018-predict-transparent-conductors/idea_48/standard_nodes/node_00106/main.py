import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys

# Import from library
from library.data import get_loaders
from library.model import BA_ADS_Model
from library.train import train_model
from library.predict import generate_predictions
from library.config import WORKING_DIR, SUBMISSION_DIR, SEED

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("Starting runfile execution...")

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to use pre-processed data if available
    # Reducing batch size slightly to ensure stability, though 32 (default) should be fine.
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=32, debug_mode=False, load_cached_data=True
    )

    # 3. Model Initialization
    model = BA_ADS_Model()
    model.to(device)

    # 4. Training
    # Limiting epochs to 50 for a fast baseline as requested.
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,  # Reduced for speed
        device=device,
    )

    # 5. Validation Assessment
    print("Performing validation assessment...")
    model.eval()

    all_preds = []
    all_targets = []
    all_globals = []

    with torch.no_grad():
        for batch_atomic, batch_index, batch_global, batch_targets, _ in val_loader:
            batch_atomic = batch_atomic.to(device)
            batch_index = batch_index.to(device)
            batch_global = batch_global.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_atomic, batch_index, batch_global)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
            all_globals.append(batch_global.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_globals = np.concatenate(all_globals, axis=0)

    # Calculate Metric
    # Targets are log1p transformed. Predictions are in log space.
    # Metric is Column-wise RMSLE.
    # Since data is already log(1+x), RMSLE is just RMSE on this data.

    mse_col_0 = mean_squared_error(all_targets[:, 0], all_preds[:, 0])
    mse_col_1 = mean_squared_error(all_targets[:, 1], all_preds[:, 1])

    rmsle_col_0 = np.sqrt(mse_col_0)
    rmsle_col_1 = np.sqrt(mse_col_1)

    # The competition metric is usually the mean of column-wise RMSLEs
    final_metric = (rmsle_col_0 + rmsle_col_1) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude (mean absolute error per sample in log space)
    errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Feature names based on get_global_features in features.py
    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Volume",
        "Aspect_Ratio_1",
        "Aspect_Ratio_2",
        "Aspect_Ratio_3",
        "Density",
        "Num_Atoms",
        "Pct_Al",
        "Pct_Ga",
        "Pct_In",
        "Mean_Mass",
        "Mean_Radius",
        "Mean_Eneg",
        "Std_Mass",
        "Std_Radius",
        "Std_Eneg",
    ]

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(all_globals, columns=feature_names)
    df_analysis["Error"] = errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["Error"].drop("Error").abs().sort_values(ascending=False)
    )

    print("Top 5 features correlated with model error:")
    print(correlations.head(5))

    # 7. Submission Generation
    threshold = 0.04819517582654953
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

        # generate_predictions handles loading test data, inference, inverse transform, and saving
        generate_predictions(
            model_path=os.path.join(WORKING_DIR, "best_model.pt"),
            output_path=submission_path,
            batch_size=32,
            device=device,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
