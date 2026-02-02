import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib

# Import library modules
import library.utils
import library.trainer
import library.data_processing
import library.dataset
import library.model

# Reload critical modules to ensure changes (like AMP and CFG) are picked up
# Cite debug_lesson_1: Reload modules when modifying configuration in persistent environments.
importlib.reload(library.utils)
importlib.reload(library.data_processing)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.trainer)

from library.utils import CFG, seed_everything
from library.data_processing import prepare_data
from library.dataset import HuBMAPDataset, get_transforms
from library.trainer import Trainer
from library.inference import generate_submission
from library.model import UNetPlusPlus


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    seed_everything(CFG.seed)

    # Adjust CFG for Optimized Solution (Cite solution_lesson_node_00014)
    CFG.epochs = 12
    CFG.batch_size = 8
    CFG.img_size = 512
    CFG.T_0 = 12

    print(
        f"Configuration: Epochs={CFG.epochs}, Batch Size={CFG.batch_size}, Image Size={CFG.img_size}, Device={CFG.device}"
    )

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("Loading metadata...")
    train_meta = pd.read_csv(CFG.train_metadata_path)
    val_meta = pd.read_csv(CFG.val_metadata_path)

    print("Preparing tile datasets (ROI constrained)...")
    # Generate/Load tiles. Overlap=0 for efficient coverage.
    train_tiles = prepare_data(
        train_meta, CFG.img_size, overlap=0, cache_dir=CFG.cache_dir, split="train"
    )
    val_tiles = prepare_data(
        val_meta, CFG.img_size, overlap=0, cache_dir=CFG.cache_dir, split="val"
    )

    print(f"Train tiles: {len(train_tiles)}, Validation tiles: {len(val_tiles)}")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader
    # ---------------------------------------------------------
    train_ds = HuBMAPDataset(
        train_tiles, transforms=get_transforms("train"), mode="train"
    )
    val_ds = HuBMAPDataset(val_tiles, transforms=get_transforms("valid"), mode="valid")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader)
    trainer.fit(epochs=CFG.epochs)

    # ---------------------------------------------------------
    # 5. Final Validation
    # ---------------------------------------------------------
    print("Running Final Validation on Hold-out Set...")

    # Load best model
    model = UNetPlusPlus(
        backbone_name=CFG.backbone, classes=CFG.num_classes, pretrained=False
    )
    checkpoint_path = os.path.join(CFG.cache_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()

    total_intersection = 0
    total_union = 0

    # Store data for failure analysis
    tile_analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(CFG.device)
            masks = batch["mask"].to(CFG.device)
            ids = batch["id"]

            # Inference (No TTA for validation speed)
            logits = model(imgs)
            preds = torch.sigmoid(logits)
            preds_bin = (preds > CFG.threshold).float()

            # Update Global Metrics
            total_intersection += (preds_bin * masks).sum().item()
            total_union += preds_bin.sum().item() + masks.sum().item()

            # Collect Per-Tile Metrics for Failure Analysis
            batch_size = imgs.size(0)
            for i in range(batch_size):
                p_flat = preds_bin[i].flatten()
                t_flat = masks[i].flatten()

                inter_i = (p_flat * t_flat).sum().item()
                union_i = p_flat.sum().item() + t_flat.sum().item()

                dice_i = (2.0 * inter_i + 1e-7) / (union_i + 1e-7)

                tile_analysis_data.append(
                    {"id": ids[i], "dice": dice_i, "error": 1.0 - dice_i}
                )

    # Compute Global Dice
    final_metric = (2.0 * total_intersection + 1e-7) / (total_union + 1e-7)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(tile_analysis_data)

    # Merge with metadata to link errors with features
    # Note: 'id' in df_analysis is the image ID.
    df_merged = pd.merge(df_analysis, val_meta, on="id", how="left")

    # Select numeric columns for correlation
    numeric_cols = [
        "age",
        "weight_kilograms",
        "height_centimeters",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
        "width_pixels",
        "height_pixels",
    ]
    # Filter columns that exist in metadata
    valid_numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

    if valid_numeric_cols:
        correlations = df_merged[valid_numeric_cols].corrwith(df_merged["error"])
        print("Correlation between Model Error (1-Dice) and Input Features:")
        print(correlations.sort_values(ascending=False))
    else:
        print("No numeric metadata features available for correlation analysis.")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    submission_threshold = 0.9132

    if final_metric > submission_threshold:
        print(
            f"\nValidation metric ({final_metric:.5f}) > {submission_threshold}. Generating submission..."
        )
        generate_submission(checkpoint_path, output_dir="./submission")
    else:
        print(
            f"\nValidation metric ({final_metric:.5f}) <= {submission_threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
