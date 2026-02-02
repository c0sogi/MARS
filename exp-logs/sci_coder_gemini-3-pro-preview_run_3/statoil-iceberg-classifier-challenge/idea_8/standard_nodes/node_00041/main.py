import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_processed_data, IcebergDataset
from library.model import SimpleCNN
from library.train import run_fold
from library.inference import create_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # Limit epochs to ensure completion within strict time limits while allowing convergence
    Config.NUM_EPOCHS = 30
    # Use full dataset as it is small (1604 samples)
    Config.MAX_SAMPLES = None

    print("Configuration set for fast baseline execution.")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Training Loop (5-Fold CV)
    # -------------------------------------------------------------------------
    print("\nStarting Training of 5 Folds...")
    for fold in range(Config.N_FOLDS):
        run_fold(fold_index=fold, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation on Hold-Out Set (Metadata)
    # -------------------------------------------------------------------------
    print("\nStarting Evaluation on Hold-Out Validation Set...")

    # Load validation metadata to identify hold-out samples
    val_meta_path = Config.VAL_META
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    val_df = pd.read_csv(val_meta_path)
    val_ids = set(val_df["id"].values)

    # Load all processed training data (cached)
    X_all, angles_all, y_all, ids_all = load_processed_data(
        is_train=True, load_cached_data=True
    )

    # Filter for validation samples
    val_indices = [i for i, uid in enumerate(ids_all) if uid in val_ids]

    if len(val_indices) == 0:
        raise ValueError("No matching validation IDs found in processed data.")

    X_val = X_all[val_indices]
    angles_val = angles_all[val_indices]
    y_val = y_all[val_indices]

    # Create DataLoader for validation
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # Perform Inference with Ensemble
    device = torch.device(Config.DEVICE)
    ensemble_probs = np.zeros(len(y_val), dtype=np.float64)

    print(f"Evaluating ensemble on {len(y_val)} validation samples...")

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}", "model_best.pth")

        # Load Model
        model = SimpleCNN()
        model.to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Test-Time Augmentation (Original + Flip)
                # Original
                logits_orig = model(images, angles)
                probs_orig = torch.sigmoid(logits_orig)

                # Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, angles)
                probs_flip = torch.sigmoid(logits_flip)

                avg_batch = (probs_orig + probs_flip) / 2.0
                fold_probs.extend(avg_batch.cpu().numpy().flatten())

        ensemble_probs += np.array(fold_probs)

    # Average across folds
    ensemble_probs /= Config.N_FOLDS

    # Calculate Metric
    final_metric = log_loss(y_val, ensemble_probs)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude
    abs_error = np.abs(ensemble_probs - y_val)

    # Calculate basic image statistics for correlation
    # X_val shape: (N, 3, 75, 75). Channel 0: HH, Channel 1: HV
    b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "abs_error": abs_error,
            "inc_angle": angles_val,
            "b1_mean": b1_mean,
            "b2_mean": b2_mean,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["abs_error"].drop("abs_error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.18145903282502943

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        create_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
