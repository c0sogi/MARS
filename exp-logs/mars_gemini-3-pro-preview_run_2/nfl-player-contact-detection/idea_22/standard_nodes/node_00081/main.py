import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from library.config import Config
from library.trainer import Trainer
from library.dataset import NFLContactDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting E-GRV-Net Pipeline...")

    # 2. Initialize Trainer
    # We use the Trainer class which encapsulates Model, Loss, and Optimizer
    trainer = Trainer()

    # 3. Train
    # We limit to 10 epochs for the fast baseline requirement.
    # The efficient tabular architecture usually converges quickly.
    print("Initiating training...")
    best_mcc = trainer.fit(epochs=10, debug=False)

    # 4. Report Metric
    # Strict requirement: Print the final validation metric in specific format
    print(f"Final Validation Metric: {best_mcc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load validation data for analysis
    val_ds = NFLContactDataset(split="validation", load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Get predictions and targets
    # trainer.validate returns avg_loss, probs, targets
    _, probs, targets = trainer.validate(val_loader)

    # Calculate Error Magnitude
    # Error = |Target - Probability|
    errors = np.abs(targets - probs).flatten()

    # Extract Key Features for Correlation
    # Features are flattened: [Lags * (P1 + P2 + Interaction)]
    # We want features at Lag 0 (Center of window)

    X_kin = val_ds.X_kin.numpy()

    W = Config.WINDOW_SIZE
    n_kin = len(Config.KINEMATIC_FEATURES)
    n_inter = len(Config.INTERACTION_FEATURES)

    # Structure per lag: [P1_Feats (n_kin), P2_Feats (n_kin), Inter_Feats (n_inter)]
    feats_per_lag = (n_kin * 2) + n_inter

    # Index of Lag 0 start
    start_idx_lag0 = W * feats_per_lag

    # Indices relative to start of Lag 0
    # Distance is the 1st interaction feature. Interaction block starts after P1 and P2 blocks.
    idx_distance = start_idx_lag0 + (n_kin * 2) + 0

    # Speed is the 3rd kinematic feature (index 2) for Player 1
    idx_speed_p1 = start_idx_lag0 + 2

    # Extract columns
    feat_distance = X_kin[:, idx_distance]
    feat_speed = X_kin[:, idx_speed_p1]

    # Calculate Correlations
    corr_dist = np.corrcoef(errors, feat_distance)[0, 1]
    corr_speed = np.corrcoef(errors, feat_speed)[0, 1]

    print(f"Correlation between Error and Distance (Lag 0): {corr_dist:.4f}")
    print(f"Correlation between Error and Speed P1 (Lag 0): {corr_speed:.4f}")

    if corr_dist < -0.1:
        print(
            "Observation: Error increases as distance decreases (Close contact ambiguity)."
        )
    elif corr_dist > 0.1:
        print(
            "Observation: Error increases as distance increases (Long range false positives)."
        )

    # 6. Submission
    threshold_score = 0.6634847318478787

    if best_mcc > threshold_score:
        print(
            f"\nValidation MCC ({best_mcc:.6f}) > Threshold ({threshold_score:.6f}). Generating submission..."
        )
        trainer.predict_and_submit()
    else:
        print(
            f"\nValidation MCC ({best_mcc:.6f}) <= Threshold ({threshold_score:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
