import os
import sys
import torch
import pandas as pd
import numpy as np

# Import required components from the provided library
from library.config import Config
from library.train import run_training
from library.inference import predict
from library.dataset import get_dataloaders
from library.model import DeepSupervisedVentilatorModel
from library.loss import MaskedL1Loss
from library.utils import seed_everything


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    # Override Config for Fast Baseline execution
    # 20 epochs is sufficient for convergence on A100 with OneCycleLR
    Config.EPOCHS = 20
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Configuration configured. Epochs: {Config.EPOCHS}, Device: {Config.DEVICE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    # run_training handles data loading, model init, training loop, and saving the best model
    run_training()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Starting Validation & Failure Analysis ===")

    device = torch.device(Config.DEVICE)

    # Load Validation DataLoader
    # We use load_cached_data=True to reuse the data processed during training
    _, val_loader, _, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Load the best model saved during training
    model = DeepSupervisedVentilatorModel().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Critical Error: Model file not found at {Config.MODEL_PATH}")
        sys.exit(1)

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Containers for analysis
    flat_preds = []
    flat_targets = []
    flat_u_out = []
    flat_u_in = []
    flat_R = []
    flat_C = []

    # Feature indices in the input tensor (based on Config.FEATURE_COLS)
    # ["time_step", "u_in", "u_out", "R", "C", ...]
    IDX_U_IN = 1
    IDX_U_OUT = 2
    IDX_R = 3
    IDX_C = 4

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            # Forward pass (in eval mode, returns only final_pred)
            preds = model(data)

            # Move to CPU for analysis
            b_preds = preds.cpu().numpy().flatten()
            b_targets = target.numpy().flatten()
            b_u_out = data[:, :, IDX_U_OUT].cpu().numpy().flatten()
            b_u_in = data[:, :, IDX_U_IN].cpu().numpy().flatten()
            b_R = data[:, :, IDX_R].cpu().numpy().flatten()
            b_C = data[:, :, IDX_C].cpu().numpy().flatten()

            flat_preds.extend(b_preds)
            flat_targets.extend(b_targets)
            flat_u_out.extend(b_u_out)
            flat_u_in.extend(b_u_in)
            flat_R.extend(b_R)
            flat_C.extend(b_C)

    # Convert to numpy arrays
    arr_preds = np.array(flat_preds)
    arr_targets = np.array(flat_targets)
    arr_u_out = np.array(flat_u_out)

    # Calculate Masked MAE (Metric)
    # Filter for inspiratory phase (u_out == 0)
    insp_mask = arr_u_out == 0

    abs_errors = np.abs(arr_preds - arr_targets)
    masked_errors = abs_errors[insp_mask]

    final_metric = np.mean(masked_errors)

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\n--- Failure Analysis ---")
    analysis_df = pd.DataFrame(
        {
            "error": abs_errors,
            "u_in": flat_u_in,
            "R": flat_R,
            "C": flat_C,
            "u_out": flat_u_out,
        }
    )

    # Correlate only on inspiratory phase
    insp_df = analysis_df[analysis_df["u_out"] == 0]
    correlations = insp_df[["error", "u_in", "R", "C"]].corr()["error"].drop("error")

    print("Correlation of Error with Features (Inspiratory Phase):")
    print(correlations)

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.3096454441547394

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        # predict() handles loading test data, model, and saving to Config.SUBMISSION_PATH
        predict(load_cached_data=True)
        print(f"Submission generation complete. Saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
