import os
import sys
import numpy as np
import pandas as pd
import torch
import random

# Import from provided library files
from library.config import Config
from library.trainer import Trainer
from library.inference import predict_test_set, create_submission_file


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline & Submission Requirements
    # Cite Lesson 00005: Extended training horizon (50 epochs) is required for convergence.
    Config.EPOCHS = 50
    Config.BATCH_SIZE = 512  # Increased for A100 efficiency
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seeds
    set_seed(Config.SEED)

    print("Configuration configured for fast baseline.")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize Trainer (loads data and model)
    # debug=False ensures we use the full validation set as required
    trainer = Trainer(load_cached_data=False, debug=False)

    # Run training
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model weights for analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
        )

    trainer.model.eval()
    val_loader = trainer.val_loader
    device = trainer.device

    # Prepare for metric calculation and failure analysis
    total_mae_sum = 0.0
    total_count = 0

    # Lists for correlation analysis
    all_errors = []
    all_feats = []

    # Load scaler params to unscale u_out and other features for analysis
    if os.path.exists(Config.SCALER_CACHE):
        scaler_params = np.load(Config.SCALER_CACHE)
        scale = scaler_params["scale"]
        center = scaler_params["center"]

        # Indices in CONT_FEATURES
        u_out_idx = Config.CONT_FEATURES.index("u_out")
        u_in_idx = Config.CONT_FEATURES.index("u_in")
        time_idx = Config.CONT_FEATURES.index("time_step")
    else:
        raise FileNotFoundError("Scaler cache not found. Cannot perform analysis.")

    with torch.no_grad():
        for batch in val_loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = trainer.model(cont, cat)

            # Unscale u_out to identify inspiratory phase
            # shape: (Batch, Seq)
            u_out_scaled = cont[:, :, u_out_idx].cpu().numpy()
            u_out = u_out_scaled * scale[u_out_idx] + center[u_out_idx]

            # Create mask: u_out == 0 is inspiratory. Use threshold 0.5
            insp_mask = u_out < 0.5

            # Calculate errors
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            abs_err = np.abs(preds_np - targets_np)

            # Aggregate Metric (MAE on Inspiratory Phase)
            if insp_mask.sum() > 0:
                masked_err = abs_err[insp_mask]
                total_mae_sum += masked_err.sum()
                total_count += insp_mask.sum()

                # Collect data for Failure Analysis
                # Recover real values for u_in and time_step
                u_in_vals = (
                    cont[:, :, u_in_idx].cpu().numpy() * scale[u_in_idx]
                    + center[u_in_idx]
                )[insp_mask]
                time_vals = (
                    cont[:, :, time_idx].cpu().numpy() * scale[time_idx]
                    + center[time_idx]
                )[insp_mask]

                # R and C indices (Categorical)
                # cat shape: (Batch, Seq, 2) -> R is idx 0, C is idx 1
                r_vals = cat[:, :, 0].cpu().numpy()[insp_mask]
                c_vals = cat[:, :, 1].cpu().numpy()[insp_mask]

                # Stack features: u_in, time_step, R_idx, C_idx
                batch_feats = np.stack([u_in_vals, time_vals, r_vals, c_vals], axis=1)

                all_errors.append(masked_err)
                all_feats.append(batch_feats)

    # Compute Final Metric
    final_metric = total_mae_sum / total_count if total_count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    if len(all_errors) > 0:
        all_errors_np = np.concatenate(all_errors)
        all_feats_np = np.concatenate(all_feats)

        df_analysis = pd.DataFrame(
            all_feats_np, columns=["u_in", "time_step", "R_idx", "C_idx"]
        )
        df_analysis["error"] = all_errors_np

        # Calculate correlation
        corrs = df_analysis.corr()["error"].drop("error")
        print("\nFailure Analysis - Correlation between Error Magnitude and Features:")
        print(corrs)

    # ==========================================
    # 4. Submission
    # ==========================================
    threshold = 0.36414578557014465

    if final_metric < threshold:
        print(f"\nMetric {final_metric:.6f} passed threshold {threshold:.6f}.")
        print("Generating submission file...")

        # Run inference on test set
        ids, preds = predict_test_set(trainer.model, trainer.test_loader, device)

        # Save submission
        create_submission_file(ids, preds, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {final_metric:.6f} did not pass threshold {threshold:.6f}.")
        print("Submission skipped.")


if __name__ == "__main__":
    main()
