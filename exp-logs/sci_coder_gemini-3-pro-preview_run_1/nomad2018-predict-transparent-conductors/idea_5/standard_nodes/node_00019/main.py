import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    VAL_METADATA_PATH,
    VAL_CACHE_PATH,
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    SEED,
)
from library.data_utils import (
    process_data,
    CrystalDataset,
    collate_fn,
)
from library.model import CADSTFModel
from library.train import run_training, generate_submission


def main():
    # 1. Train the model
    # We limit epochs to 50 for a fast baseline execution as per requirements.
    # The run_training function handles data loading, scaling, and training loop.
    print(">>> Starting Training Pipeline")
    scaler = run_training(max_samples=None, num_epochs=50)

    # 2. Load Validation Data for Assessment
    print("\n>>> Loading Validation Data")
    val_data_dict = process_data(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data=True
    )

    # Create validation dataset and loader
    # Note: We must use the scaler fitted during training
    val_dataset = CrystalDataset(val_data_dict, scaler=scaler)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    # 3. Load Best Model
    print(">>> Loading Best Model for Inference")
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")

    model = CADSTFModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 4. Run Validation Inference
    print(">>> Running Validation Inference")
    all_preds_log = []
    all_targets_log = []

    # We collect global features for failure analysis
    # Note: These should be unscaled for interpretability, so we take them from the raw dict
    # However, the dataset object has scaled them. We will use the raw numpy array from val_data_dict.
    raw_global_features = val_data_dict["global_features"]

    with torch.no_grad():
        for batch in val_loader:
            global_feats = batch["global_features"].to(DEVICE)
            atomic_feats = batch["atomic_features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            outputs = model(global_feats, atomic_feats, mask)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())

    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)

    # 5. Compute Metric
    # Metric: Column-wise root mean squared logarithmic error.
    # Since inputs/outputs are already log(1+x), we just calculate RMSE on these values.
    mse_per_col = np.mean((all_preds_log - all_targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n>>> Performing Failure Analysis")
    # Calculate error magnitude per sample (mean squared error in log space)
    sample_errors = np.mean((all_preds_log - all_targets_log) ** 2, axis=1)

    # Feature names based on data_utils.process_data order:
    # [len_a, len_b, len_c, ang_a, ang_b, ang_c, comp_Al, comp_Ga, comp_In, vol, dens]
    feature_names = [
        "lattice_len_a",
        "lattice_len_b",
        "lattice_len_c",
        "lattice_ang_alpha",
        "lattice_ang_beta",
        "lattice_ang_gamma",
        "comp_Al",
        "comp_Ga",
        "comp_In",
        "volume",
        "density",
    ]

    # Create DataFrame for correlation analysis
    analysis_df = pd.DataFrame(raw_global_features, columns=feature_names)
    analysis_df["error_magnitude"] = sample_errors

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    print("Correlation between Input Features and Error Magnitude:")
    print(correlations.sort_values(ascending=False))

    # 7. Generate Submission
    # Threshold defined in requirements
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"\n>>> Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(scaler, max_samples=None)
    else:
        print(
            f"\n>>> Metric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
