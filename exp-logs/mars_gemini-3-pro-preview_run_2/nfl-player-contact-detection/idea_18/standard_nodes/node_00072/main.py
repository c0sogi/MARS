import os
import sys
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calc_mcc
from library.train_eval import train_model, generate_submission, validate
from library.dataset import get_dataloaders
from library.models import RVCNet


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override defaults for a fast baseline execution
    Config.EPOCHS = 8
    Config.BATCH_SIZE = 8192  # Leverage A100 memory

    seed_everything(Config.SEED)
    print(
        f"Starting pipeline with EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n=== Training Model ===")
    # train_model handles data loading, training, and saving the best model/threshold
    best_threshold = train_model(load_cached_data=True)
    print(f"Training finished. Best threshold: {best_threshold:.4f}")

    # =========================================================================
    # 3. Validation Assessment
    # =========================================================================
    print("\n=== Final Validation Assessment ===")
    device = torch.device(Config.DEVICE)

    # Load the best model artifact
    model = RVCNet().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {Config.MODEL_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Get validation data loader
    # We use the same batch size as training for efficiency
    _, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Run inference
    val_logits, val_targets = validate(model, val_loader, device)

    # Calculate probabilities and predictions
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_preds = (val_probs > best_threshold).astype(int)

    # Calculate and print final metric
    final_mcc = calc_mcc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_mcc}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (absolute difference)
    errors = np.abs(val_probs - val_targets)

    # Extract features from the loader to correlate with errors
    # The loader returns batches, so we accumulate them
    k_feats_list = []
    v_feats_list = []

    for (k_batch, v_batch), _ in val_loader:
        k_feats_list.append(k_batch.numpy())
        v_feats_list.append(v_batch.numpy())

    X_kin = np.concatenate(k_feats_list, axis=0)
    X_vis = np.concatenate(v_feats_list, axis=0)
    X_all = np.hstack([X_kin, X_vis])

    # Reconstruct feature names to match the column order defined in data_processing.py
    kin_names = []
    vis_names = []

    for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
        suffix = f"_lag_{k}"
        # Kinematic columns order
        for c in Config.KINEMATIC_PLAYER_COLS:
            kin_names.append(f"Kin_{c}{suffix}_p1")
        for c in Config.KINEMATIC_PLAYER_COLS:
            kin_names.append(f"Kin_{c}{suffix}_p2")
        for c in Config.KINEMATIC_PAIR_COLS:
            kin_names.append(f"Kin_{c}{suffix}")

        # Visual columns order
        for c in Config.VISUAL_PLAYER_COLS:
            vis_names.append(f"Vis_{c}{suffix}_p1")
        for c in Config.VISUAL_PLAYER_COLS:
            vis_names.append(f"Vis_{c}{suffix}_p2")
        for c in Config.VISUAL_PAIR_COLS:
            vis_names.append(f"Vis_{c}{suffix}")

    all_names = kin_names + vis_names

    # Calculate correlations between features and error
    print("Calculating feature correlations with error magnitude...")

    # Center the error vector
    error_mean = np.mean(errors)
    error_centered = errors - error_mean
    error_std = np.std(errors)

    correlations = []
    if error_std > 1e-9:
        # Center features
        X_mean = np.mean(X_all, axis=0)
        X_centered = X_all - X_mean
        X_std = np.std(X_all, axis=0)

        # Identify valid columns (non-constant)
        valid_cols = X_std > 1e-9

        # Vectorized covariance calculation
        cov = np.mean(X_centered[:, valid_cols] * error_centered[:, None], axis=0)
        corr = cov / (X_std[valid_cols] * error_std)

        # Map correlations back to feature names
        valid_indices = np.where(valid_cols)[0]
        for idx, c_val in zip(valid_indices, corr):
            correlations.append((all_names[idx], c_val))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Model Error:")
    for name, val in correlations[:10]:
        print(f"  {name}: {val:.4f}")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    TARGET_METRIC = 0.625814796769196

    if final_mcc > TARGET_METRIC:
        print(f"\nValidation Metric ({final_mcc}) exceeds target ({TARGET_METRIC}).")
        print("Generating submission file...")
        generate_submission(threshold=best_threshold)
    else:
        print(
            f"\nValidation Metric ({final_mcc}) does not exceed target ({TARGET_METRIC})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
