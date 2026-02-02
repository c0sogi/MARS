import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.model import DPSCACNN
from library.trainer import run_fold
from library.inference import predict_test
from library.data_loader import IcebergDataset


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Configuration Override for Fast Baseline
    # Reducing epochs to 25 ensures the script completes quickly while being sufficient for this small dataset.
    Config.NUM_EPOCHS = 25
    Config.NUM_FOLDS = 5

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(
        f"Starting execution with {Config.NUM_FOLDS} folds and {Config.NUM_EPOCHS} epochs per fold."
    )

    # 2. Train Folds
    # We train all 5 folds to create a robust ensemble.
    for fold in range(Config.NUM_FOLDS):
        run_fold(fold, load_cached_data=True)

    # 3. Validation on Hold-out Set
    # We explicitly load the validation set defined in metadata/val.csv as requested.
    print("Loading hold-out validation set for evaluation...")

    val_meta_path = Config.VAL_META_PATH
    if not os.path.exists(val_meta_path):
        print(f"Error: Metadata file not found at {val_meta_path}")
        return

    val_meta = pd.read_csv(val_meta_path)
    val_ids = set(val_meta["id"].values)

    # Load raw train.json to extract the specific validation samples
    with open(Config.TRAIN_JSON, "r") as f:
        raw_data = json.load(f)

    # Filter for validation samples
    val_samples = [item for item in raw_data if item["id"] in val_ids]
    df_val = pd.DataFrame(val_samples)

    if len(df_val) == 0:
        print("Error: No validation samples found.")
        return

    # Preprocess Validation Images
    # Band 1 (HH)
    x_band1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df_val["band_1"]]
    )
    # Band 2 (HV)
    x_band2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df_val["band_2"]]
    )
    # Band 3 (Avg)
    x_band3 = (x_band1 + x_band2) / 2.0

    # Stack to (N, 3, 75, 75)
    X_val = np.stack((x_band1, x_band2, x_band3), axis=1)

    # Preprocess Angles
    # Impute missing angles with median (using local median for simplicity in this baseline)
    angles = pd.to_numeric(df_val["inc_angle"], errors="coerce")
    median_angle = angles.median()
    if pd.isna(median_angle):
        median_angle = 0.0  # Fallback
    ang_val = angles.fillna(median_angle).astype(np.float32).values

    # Targets
    y_val = df_val["is_iceberg"].values.astype(np.float32)

    # Create DataLoader
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. Ensemble Inference on Validation Set
    print("Running ensemble inference on validation set...")
    ensemble_preds = np.zeros(len(y_val))

    for fold in range(Config.NUM_FOLDS):
        model = DPSCACNN().to(device)
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint for fold {fold} not found. Skipping.")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        ensemble_preds += np.array(fold_preds)

    # Average predictions
    ensemble_preds /= Config.NUM_FOLDS

    # 5. Metric Calculation
    # Clip predictions to prevent log(0)
    ensemble_preds_clipped = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_val, ensemble_preds_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - ensemble_preds)

    # Compute simple image statistics for correlation
    b1_mean = np.mean(x_band1, axis=(1, 2))
    b2_mean = np.mean(x_band2, axis=(1, 2))
    b1_std = np.std(x_band1, axis=(1, 2))
    b2_std = np.std(x_band2, axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_val,
            "b1_mean": b1_mean,
            "b2_mean": b2_mean,
            "b1_std": b1_std,
            "b2_std": b2_std,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
