import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.train import train_model
from library.predict import generate_submission
from library.dataset import get_datasets, collate_fn
from library.model import APDeepSets


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Orchestration running on device: {device}")

    # 2. Train Model
    # We use 50 epochs for a fast baseline. The dataset is small enough that this runs quickly.
    print("\n--- Starting Training ---")
    train_model(debug=False, num_epochs=50)

    # 3. Validation Assessment
    print("\n--- Starting Validation ---")
    _, val_dataset, _ = get_datasets(debug=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Load the best model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model = APDeepSets()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            global_features = batch["global_features"].to(device)
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            model_input = {
                "global_features": global_features,
                "atomic_features": atomic_features,
                "batch_indices": batch_indices,
            }

            outputs = model(model_input)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_global_feats.append(global_features.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    global_feats = np.concatenate(all_global_feats, axis=0)

    # 4. Metric Calculation
    # Targets and Preds are already in log1p space.
    # Metric is Column-wise RMSLE.
    # RMSLE(y, pred) = RMSE(log1p(y), log1p(pred))
    # Since our model outputs log1p(pred) and targets are log1p(y), we just calculate RMSE.

    mse_col1 = mean_squared_error(targets[:, 0], preds[:, 0])
    mse_col2 = mean_squared_error(targets[:, 1], preds[:, 1])

    rmse_col1 = np.sqrt(mse_col1)
    rmse_col2 = np.sqrt(mse_col2)

    final_metric = (rmse_col1 + rmse_col2) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (L1 error in log space)
    errors = np.abs(targets - preds).mean(
        axis=1
    )  # Average error across the two targets per sample

    # Feature names based on data_utils.process_tabular_features
    feature_names = [
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
        "volume",
        "density",
    ]

    # Create a DataFrame for correlation analysis
    # Note: global_feats might be normalized, but correlation is invariant to linear scaling
    analysis_df = pd.DataFrame(global_feats, columns=feature_names)
    analysis_df["error"] = errors

    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )

    print("Correlation between Error Magnitude and Global Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = 0.05781995991591556
    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} is lower than threshold {threshold}. Generating submission..."
        )
        generate_submission(debug=False)
    else:
        print(
            f"\nMetric {final_metric} is NOT lower than threshold {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
