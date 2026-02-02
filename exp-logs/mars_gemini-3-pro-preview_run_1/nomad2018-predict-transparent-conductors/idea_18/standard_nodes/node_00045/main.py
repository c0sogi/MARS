import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.train import run_training, set_seed
from library.data import CrystalDataset, collate_crystals
from library.model import CRNDSModel
from library.inference import generate_submission


def main():
    # 1. Train the model
    # The configuration is set to 200 epochs with early stopping, which fits within the time limit.
    print("Starting Training Pipeline...")
    run_training()

    # 2. Validation & Metric Calculation
    print("\nStarting Validation...")
    device = torch.device(Config.DEVICE)

    # Load validation data (using cached scalers from training)
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_META_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="val",
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=0,
    )

    # Load the best model saved during training
    model = CRNDSModel().to(device)
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
        )

    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, batch_index)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            # Store global features for failure analysis (move to CPU)
            all_global_feats.append(batch["global_features"].numpy())

    preds_log = np.vstack(all_preds_log)
    targets_log = np.vstack(all_targets_log)
    global_feats = np.vstack(all_global_feats)

    # Calculate Column-wise RMSLE
    # Since model predicts log(1+y) and targets are log(1+y),
    # RMSE on these values is equivalent to RMSLE on original scale.
    rmse_col1 = np.sqrt(mean_squared_error(targets_log[:, 0], preds_log[:, 0]))
    rmse_col2 = np.sqrt(mean_squared_error(targets_log[:, 1], preds_log[:, 1]))
    final_metric = (rmse_col1 + rmse_col2) / 2

    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Calculate mean absolute error per sample across targets
    errors = np.mean(np.abs(preds_log - targets_log), axis=1)

    # Feature names corresponding to the order in library/data.py
    feature_names = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "cell_volume",
        "atomic_density",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "number_of_total_atoms",
    ]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(global_feats, columns=feature_names)
    analysis_df["error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )
    print(correlations.head(5))

    # 4. Submission
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission()
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
