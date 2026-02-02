import os
import torch
import numpy as np
import pandas as pd

# Import provided library components
from library.config import set_seed, DEVICE, CACHE_DIR, VAL_CSV, BATCH_SIZE, TARGET_COLS
from library.data_loader import get_dataloaders
from library.model import SpectrogramCRNN
from library.trainer import Trainer
from library.inference import predict_and_submit
from library.utils import compute_kl_divergence


def run_pipeline():
    # 1. Setup and Initialization
    print("--- Initializing Pipeline ---")
    set_seed(42)

    # 2. Data Loading
    # Load cached data to speed up initialization
    print("\n[Data] Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        load_cached_data=True,
    )

    if train_loader is None or val_loader is None:
        print("Critical Error: Failed to load data.")
        return

    # 3. Model Initialization
    print("\n[Model] Initializing SpectrogramCRNN...")
    model = SpectrogramCRNN()
    model.to(DEVICE)

    # 4. Training
    # We limit training to 5 epochs to ensure the script completes quickly as a baseline.
    print("\n[Training] Starting training loop (Fast Baseline)...")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE)
    trainer.fit(epochs=5)

    # 5. Validation Assessment
    print("\n[Validation] Performing final evaluation on hold-out set...")
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    # Load the best weights saved by the trainer
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Warning: Best model weights not found. Using current model state.")

    model.eval()

    val_preds = []
    val_targets = []

    # Inference on validation set (No Grad for speed/memory)
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(DEVICE)
            outputs = model(data)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    # Compute Final Metric
    final_metric = compute_kl_divergence(val_targets, val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n[Analysis] Running Failure Analysis...")
    try:
        val_df = pd.read_csv(VAL_CSV)

        # Verify alignment between metadata and predictions
        if len(val_df) == len(val_preds):
            # Calculate per-sample KL Divergence for analysis
            # Formula: sum(P * log(P/Q))
            epsilon = 1e-15
            y_pred_safe = np.clip(val_preds, epsilon, 1 - epsilon)
            # Re-normalize just in case
            y_pred_safe = y_pred_safe / np.sum(y_pred_safe, axis=1, keepdims=True)

            term1 = np.zeros_like(val_targets)
            mask = val_targets > 0
            term1[mask] = val_targets[mask] * np.log(val_targets[mask])
            term2 = val_targets * np.log(y_pred_safe)

            # KL per sample
            kl_errors = np.sum(term1 - term2, axis=1)
            val_df["error_kl"] = kl_errors

            # Define features to correlate with error
            # We look at offsets and vote counts (confidence proxies)
            feature_cols = [
                "eeg_label_offset_seconds",
                "spectrogram_label_offset_seconds",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]

            # Add total votes (annotator count/consensus strength proxy)
            vote_cols = [c for c in feature_cols if "vote" in c]
            val_df["total_votes"] = val_df[vote_cols].sum(axis=1)
            feature_cols.append("total_votes")

            # Calculate correlations
            correlations = (
                val_df[feature_cols + ["error_kl"]].corr()["error_kl"].drop("error_kl")
            )

            print("Correlation between Error Magnitude (KL) and Input Features:")
            print(correlations.sort_values(ascending=False))

            # Additional insight: Error by Expert Consensus Class
            if "expert_consensus" in val_df.columns:
                print("\nMean Error by Expert Consensus Class:")
                print(
                    val_df.groupby("expert_consensus")["error_kl"]
                    .mean()
                    .sort_values(ascending=False)
                )
        else:
            print(
                f"Skipping detailed analysis: Metadata rows ({len(val_df)}) != Prediction rows ({len(val_preds)})"
            )

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # 7. Submission
    print("\n[Submission] Generating predictions for Test Set...")
    predict_and_submit(model_path=best_model_path, device=DEVICE)

    print("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    run_pipeline()
