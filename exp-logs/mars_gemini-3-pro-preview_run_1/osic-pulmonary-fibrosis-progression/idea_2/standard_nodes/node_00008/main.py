import sys
import os
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric_numpy
from library.data import PulmonaryDataset
from library.model import TriSlabModel
from library.train import run_training
from library.inference import generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Optimize for fast baseline execution: Reduce epochs
    # Cite solution_lesson_node_00007: Increased epochs for convergence with augmentation
    Config.EPOCHS = 30

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("Starting Training...")
    run_training()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation Analysis...")

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    model = TriSlabModel(Config)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    # We read the raw CSV, but PulmonaryDataset handles the preprocessing (merging baseline info)
    val_df_raw = pd.read_csv(Config.VAL_CSV)
    val_dataset = PulmonaryDataset(val_df_raw, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    all_preds = []
    all_targets = []
    all_base_fvc = []
    all_time_delta = []

    with torch.no_grad():
        for imgs, tabular, base_fvc, time_delta, targets in val_loader:
            imgs = imgs.to(device)
            tabular = tabular.to(device)

            preds = model(imgs, tabular)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_base_fvc.append(base_fvc.cpu().numpy())
            all_time_delta.append(time_delta.cpu().numpy())

    # Concatenate results
    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    base_fvc_np = np.concatenate(all_base_fvc)
    time_delta_np = np.concatenate(all_time_delta)

    # Calculate Metric
    final_metric = calculate_metric_numpy(
        preds_np, targets_np, base_fvc_np, time_delta_np
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Absolute Error
    # FVC_pred = Base + alpha * t
    alpha = preds_np[:, 0]
    fvc_pred = base_fvc_np + alpha * time_delta_np
    abs_error = np.abs(targets_np - fvc_pred)

    # Get features from the processed dataset dataframe to ensure alignment
    analysis_df = val_dataset.df.copy()
    analysis_df["AbsError"] = abs_error

    # Encode categoricals for correlation analysis
    # Sex: Male=0, Female=1
    analysis_df["Sex_Enc"] = analysis_df["Sex"].apply(lambda x: 0 if x == "Male" else 1)

    # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
    def encode_smoking(x):
        if x == "Ex-smoker":
            return 0
        if x == "Never smoked":
            return 1
        return 2

    analysis_df["Smoking_Enc"] = analysis_df["SmokingStatus"].apply(encode_smoking)

    # Select columns for correlation
    # We include Weeks (relative time) as it's a key factor in error accumulation
    corr_cols = ["Age", "Percent", "Weeks", "Sex_Enc", "Smoking_Enc", "AbsError"]

    correlations = analysis_df[corr_cols].corr()["AbsError"].drop("AbsError")

    print("\nFailure Analysis - Feature Correlations with Absolute Error:")
    print(correlations)

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    # Threshold check
    THRESHOLD = -6.661946993018998

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
