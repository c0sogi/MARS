import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.engine import Trainer
from library.inference import InferenceRunner
from library.data import prepare_tiles, HuBMAPDataset, get_transforms
from library.model import MultiTaskResNetFPN


def main():
    # ====================================================
    # 1. Configuration & Setup
    # ====================================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 1
    # We use the full dataset (10 images) for 1 epoch as it fits within the time limit on A100
    Config.DEBUG = False

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Debug={Config.DEBUG}, Device={Config.DEVICE}"
    )

    # ====================================================
    # 2. Training
    # ====================================================
    print("\n--- Starting Training ---")
    trainer = Trainer()
    trainer.fit(load_cached_data=True)

    # ====================================================
    # 3. Validation & Metric Calculation
    # ====================================================
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load validation metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    val_df = pd.read_csv(val_meta_path)

    # Prepare tiles (using validation mode)
    val_tiles = prepare_tiles(val_df, mode="val", load_cached_data=True)

    # Dataset & Loader
    val_dataset = HuBMAPDataset(
        val_tiles, transforms=get_transforms(mode="val"), mode="val"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = MultiTaskResNetFPN()
    model.to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded best model weights from {Config.MODEL_PATH}")
    else:
        print("Warning: Best model file not found. Using current model weights.")

    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []
    tile_errors = []  # Store (id, error) for failure analysis

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            # Forward pass
            outputs = model(images)

            # Get Primary Head (Channel 0) and apply sigmoid
            preds = torch.sigmoid(outputs[:, 0, :, :])
            targets = masks[:, 0, :, :]

            # Move to CPU for metric calculation
            preds_cpu = preds.cpu()
            targets_cpu = targets.cpu()

            all_preds.append(preds_cpu)
            all_targets.append(targets_cpu)

            # Calculate error per tile for failure analysis
            for j in range(preds_cpu.shape[0]):
                p = preds_cpu[j]
                t = targets_cpu[j]

                # Binarize
                p_bin = (p > Config.MASK_THRESHOLD).float()

                # Dice calculation per tile
                intersection = (p_bin * t).sum()
                dice = (2.0 * intersection + 1e-6) / (p_bin.sum() + t.sum() + 1e-6)
                error = 1.0 - dice.item()

                # Map back to image ID
                dataset_idx = i * Config.BATCH_SIZE + j
                if dataset_idx < len(val_dataset):
                    img_id = val_dataset.tiles[dataset_idx]["id"]
                    tile_errors.append({"id": img_id, "error": error})

    # Compute Global Dice Metric
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Binarize global predictions
    all_preds_bin = (all_preds > Config.MASK_THRESHOLD).float()

    # Calculate Global Dice
    intersection = (all_preds_bin * all_targets).sum()
    final_metric = (2.0 * intersection + 1e-6) / (
        all_preds_bin.sum() + all_targets.sum() + 1e-6
    )
    final_metric = final_metric.item()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # ====================================================
    # 4. Failure Analysis
    # ====================================================
    print("\n--- Failure Analysis ---")
    if tile_errors:
        error_df = pd.DataFrame(tile_errors)
        # Aggregate error by image ID
        img_errors = error_df.groupby("id")["error"].mean().reset_index()

        # Merge with metadata
        analysis_df = pd.merge(val_df, img_errors, on="id", how="inner")

        # Identify numeric columns
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "error"]

        print("Correlation between Error and Input Features:")
        for col in numeric_cols:
            if analysis_df[col].nunique() > 1:  # Skip constant columns
                # Drop NaNs for correlation
                valid_data = analysis_df[[col, "error"]].dropna()
                if len(valid_data) > 1:
                    corr = valid_data[col].corr(valid_data["error"])
                    print(f"  {col}: {corr:.4f}")
    else:
        print("No tile errors recorded.")

    # ====================================================
    # 5. Submission
    # ====================================================
    THRESHOLD = 0.9235877394676208

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        inference_runner = InferenceRunner()
        inference_runner.run()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
