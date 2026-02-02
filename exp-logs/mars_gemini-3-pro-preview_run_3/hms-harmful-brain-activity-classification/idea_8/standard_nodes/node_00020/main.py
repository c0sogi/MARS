import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence
from library.data import get_dataloader
from library.model import AttentiveDualScaleNetwork
from library.train import run_training
from library.inference import predict


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline run
    Config.EPOCHS = 2

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("=" * 40)
    print("Starting Fast Baseline Execution")
    print("=" * 40)

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n[Step 1] Training Model...")
    # run_training handles the training loop, validation monitoring, and model saving.
    # We use load_cached_data=True to utilize any pre-existing .npy files in working dir.
    run_training(debug=False, load_cached_data=True)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n[Step 2] Validating and Computing Metric...")
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = AttentiveDualScaleNetwork()
    model.to(device)

    # Load Best Weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model weights from {Config.MODEL_PATH}")
    else:
        print("Error: Model weights not found. Training may have failed.")
        sys.exit(1)

    model.eval()

    # Get Validation Loader
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            x_eeg, x_spec = inputs
            x_eeg = x_eeg.to(device)
            x_spec = x_spec.to(device)

            # Forward pass
            logits = model((x_eeg, x_spec))
            probs = F.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute Metric
    final_metric = kl_divergence(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n[Step 3] Performing Failure Analysis...")

    # Calculate per-sample KL Divergence for correlation analysis
    epsilon = 1e-15
    y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true_safe = np.clip(y_true, epsilon, 1.0)

    # KL = sum(p * (log(p) - log(q)))
    per_sample_error = np.sum(
        y_true * (np.log(y_true_safe) - np.log(y_pred_safe)), axis=1
    )

    # Load Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    if len(val_df) == len(per_sample_error):
        val_df["kl_error"] = per_sample_error

        # Select numerical features for correlation
        features_to_check = [
            "total_votes",
            "eeg_label_offset_seconds",
            "spectogram_label_offset_seconds",
        ]
        # Filter features that exist in the dataframe
        features_to_check = [f for f in features_to_check if f in val_df.columns]

        # Compute Correlation
        correlations = (
            val_df[features_to_check + ["kl_error"]].corr()["kl_error"].drop("kl_error")
        )

        print("Correlation between Error Magnitude and Metadata Features:")
        print(correlations)
    else:
        print(
            f"Warning: Mismatch between validation set size ({len(val_df)}) and predictions ({len(per_sample_error)}). Skipping detailed failure analysis."
        )

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n[Step 4] Checking Submission Criteria...")
    THRESHOLD = 0.9081844091415405

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        submission_df = predict(debug=False, load_cached_data=True)

        # Save to required location
        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "submission.csv")

        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
