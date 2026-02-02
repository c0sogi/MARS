import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, rle_decode, dice_coef
from library.dataset import HuBMAPDataset
from library.model import FPNResNet34
from library.trainer import Trainer
from library.inference import InferenceRunner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Define paths
    train_metadata_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    val_metadata_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")

    # 2. Data Loading
    # Load metadata
    if not os.path.exists(train_metadata_path) or not os.path.exists(val_metadata_path):
        print("Error: Metadata files not found.")
        return

    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)

    # Initialize Datasets
    # Enable caching to speed up tile generation
    train_dataset = HuBMAPDataset(train_df, split="train", load_cached_data=True)
    val_dataset = HuBMAPDataset(val_df, split="val", load_cached_data=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Training
    # Initialize Model
    model = FPNResNet34(num_classes=Config.CLASSES)

    # Initialize Trainer
    trainer = Trainer(model)

    # Run Training
    # Use Config.NUM_EPOCHS and implement warmup to prevent premature stopping
    # Cite {solution_lesson_node_00009}
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=5,
        warmup_epochs=10,
    )

    # 4. Validation & Failure Analysis
    print("\nRunning full-image validation and failure analysis...")

    # Load the best model checkpoint for analysis
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found.")
        return

    # Initialize Inference Runner
    inference_runner = InferenceRunner(checkpoint_path)

    val_results = []

    # Iterate over validation images to compute full-image Dice
    for idx, row in val_df.iterrows():
        img_id = row["id"]
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        gt_rle = row["encoding"]
        h, w = int(row["height_pixels"]), int(row["width_pixels"])

        # Predict mask using sliding window inference
        try:
            pred_rle = inference_runner.predict_large_image(img_path)

            # Decode masks
            pred_mask = rle_decode(pred_rle, (h, w))
            gt_mask = rle_decode(gt_rle, (h, w))

            # Compute Dice
            score = dice_coef(pred_mask, gt_mask)

            # Record results
            res = row.to_dict()
            res["dice"] = score
            res["error"] = 1.0 - score
            val_results.append(res)

        except Exception as e:
            print(f"Error validating image {img_id}: {e}")

    # Compute and print Final Validation Metric
    if val_results:
        val_results_df = pd.DataFrame(val_results)
        final_metric = val_results_df["dice"].mean()
        print(f"Final Validation Metric: {final_metric}")

        # Failure Analysis: Correlation
        print("\nFailure Analysis (Correlation of Error with Metadata):")
        numerical_cols = [
            "age",
            "weight_kilograms",
            "height_centimeters",
            "bmi_kg/m^2",
            "percent_cortex",
            "percent_medulla",
        ]

        # Select columns that exist in the dataframe and have valid data
        cols_to_analyze = [c for c in numerical_cols if c in val_results_df.columns]

        if cols_to_analyze:
            # Compute correlation between metadata features and error (1 - Dice)
            # Drop NaN values to avoid errors in correlation calculation
            analysis_df = val_results_df[cols_to_analyze + ["error"]].dropna()

            if len(analysis_df) > 1:
                correlations = analysis_df.corr()["error"].drop("error")
                print(correlations)
            else:
                print("Not enough validation samples for correlation analysis.")
        else:
            print("No matching numerical columns found for analysis.")
    else:
        print("No validation results generated.")

    # 5. Submission Generation
    if final_metric > 0.8873:
        print("\nGenerating submission...")
        inference_runner.generate_submission()
    else:
        print(
            f"\nValidation metric {final_metric} did not meet threshold 0.8873. Skipping submission."
        )

    print("Process completed.")


if __name__ == "__main__":
    main()
