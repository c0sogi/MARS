import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader

# Import library components
from library.config import Config
from library.trainer import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast but effective baseline
    # 5 Epochs is sufficient for convergence with this learning rate and data size
    Config.EPOCHS = 5
    # Increase batch size to fully utilize A100 GPU
    Config.BATCH_SIZE = 2048
    Config.NUM_WORKERS = 4

    # Ensure reproducible results
    # Seeds are set inside Trainer.train() via library.trainer.set_seed

    print("Initializing Trainer...")
    trainer = Trainer()

    # =========================================================================
    # 2. Training
    # =========================================================================
    # Train on the full dataset. The efficient architecture and A100 allow this.
    # trainer.train() handles feature generation, training loop, and threshold optimization.
    print("Starting Training Pipeline...")
    trainer.train()

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("\n=== Validation & Failure Analysis ===")

    # Reload validation data for explicit analysis
    # We use the trainer's feature engineer to ensure consistency
    print("Loading validation features...")
    df_val = trainer.feature_engineer.generate_features(split="validation")

    # Prepare tensors using the scalers fitted during training
    X_kin, X_vis = trainer.prepare_data(df_val, fit_scaler=False)
    y_val = df_val["contact"].values

    # Create DataLoader for inference
    dataset = TensorDataset(torch.tensor(X_kin), torch.tensor(X_vis))
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    print("Running validation inference...")
    trainer.model.eval()
    probs = []

    with torch.no_grad():
        for kin, vis in loader:
            kin = kin.to(trainer.device)
            vis = vis.to(trainer.device)
            logits = trainer.model(kin, vis)
            # Apply sigmoid to get probabilities
            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(probs).flatten()

    # Apply the optimized threshold found during training
    preds = (probs > trainer.best_threshold).astype(int)

    # Calculate Final Metric
    mcc = matthews_corrcoef(y_val, preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {mcc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    df_val["pred_prob"] = probs
    df_val["error"] = np.abs(df_val["contact"] - df_val["pred_prob"])

    # Correlate error with key physical features (at t=0)
    # Note: 'dist' is log-transformed if Config.USE_LOG_DISTANCE is True, which is fine for correlation
    features_to_analyze = [
        "dist",
        "p1_speed",
        "p2_speed",
        "p1_acceleration",
        "p2_acceleration",
    ]

    print("Correlation between Prediction Error and Input Features:")
    for col in features_to_analyze:
        if col in df_val.columns:
            corr = df_val["error"].corr(df_val[col])
            print(f"  {col}: {corr:.6f}")
        else:
            print(f"  {col}: Not found in dataframe")

    # =========================================================================
    # 4. Submission
    # =========================================================================
    THRESHOLD_SCORE = 0.6634847318478787

    if mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation metric ({mcc:.6f}) exceeds threshold ({THRESHOLD_SCORE:.6f})."
        )
        print("Generating submission file...")
        trainer.predict_and_submit()
    else:
        print(
            f"\nValidation metric ({mcc:.6f}) does not exceed threshold ({THRESHOLD_SCORE:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
