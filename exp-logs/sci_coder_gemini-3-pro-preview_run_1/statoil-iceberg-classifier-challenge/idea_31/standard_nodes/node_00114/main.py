import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Import library components
from library.config import Config
from library.pipeline import Pipeline
from library.data import process_and_cache_data, IcebergDataset, get_transforms
from library.model import IsovariantResNet18
from library.engine import predict_tta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration for Fast Baseline
    # =========================================================================
    # Adjust epochs to ensure execution within time limits while maintaining performance
    Config.MAX_EPOCHS_PHASE1 = 20
    Config.SWA_EPOCHS = 5
    Config.EARLY_STOPPING_PATIENCE = 5
    Config.SCHEDULER_PATIENCE = 3

    # Ensure working directories exist
    Config.setup_dirs()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Pipeline Execution
    # =========================================================================
    pipeline = Pipeline()

    # Phase 1: Calibration (Find optimal epochs)
    print("\n=== Phase 1: Calibration ===")
    e_opt = pipeline.run_calibration_phase()

    # Phase 2: Production (Train Ensemble)
    print("\n=== Phase 2: Production ===")
    pipeline.run_production_phase(e_opt)

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("\n=== Validation & Failure Analysis ===")

    # Load Validation Data
    # We load the full processed train cache and filter by validation IDs
    data = process_and_cache_data("train", load_cached_data=True)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    val_ids_set = set(val_meta["id"].values)

    # Filter for validation samples
    # Create a boolean mask
    mask = np.array([id_ in val_ids_set for id_ in data["ids"]])

    val_images = data["images"][mask]
    val_angles = data["angles"][mask]
    val_labels = data["labels"][mask]

    print(f"Validation Set Size: {len(val_labels)}")

    # Create Validation Dataset and Loader
    val_ds = IcebergDataset(
        val_images,
        val_angles,
        val_labels,
        transform=get_transforms("val"),  # No augmentation for inference
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load SWA Models
    models = []
    for i in range(5):
        path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
        if os.path.exists(path):
            model = IsovariantResNet18().to(device)
            state_dict = torch.load(path, map_location=device)

            # Handle state_dict keys (remove 'module.' prefix if present from AveragedModel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k.startswith("n_averaged"):
                    continue
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            model.eval()
            models.append(model)

    if not models:
        print("Error: No models found for validation.")
        return

    # Ensemble Inference
    ensemble_preds = []
    for model in models:
        # predict_tta returns probabilities (sigmoid applied)
        preds = predict_tta(model, val_loader, device)
        ensemble_preds.append(preds)

    # Average Predictions
    avg_preds = np.mean(ensemble_preds, axis=0)

    # Compute Metric (Log Loss)
    # Clip predictions to avoid log(0) errors
    avg_preds_clipped = np.clip(avg_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_labels, avg_preds_clipped)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(val_labels - avg_preds)

    # Correlation between Error and Incidence Angle
    corr, _ = pearsonr(errors, val_angles)
    print(f"Correlation between Error and Incidence Angle: {corr}")

    # =========================================================================
    # 4. Submission
    # =========================================================================
    threshold = 0.16918645240183008
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({threshold}). Generating submission..."
        )
        pipeline.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
