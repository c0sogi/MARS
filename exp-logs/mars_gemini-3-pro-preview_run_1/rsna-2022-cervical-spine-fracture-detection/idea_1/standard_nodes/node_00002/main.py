import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.train import run_training, seed_everything, prepare_validation_slice_df
from library.dataset import (
    FractureSliceDataset,
    get_transforms,
)
from library.model import FractureClassifier
from library.utils import weighted_log_loss
from library.inference import run_inference


def perform_failure_analysis(df_results, val_slice_df):
    """
    Analyzes model performance to identify error patterns.
    Args:
        df_results (pd.DataFrame): DataFrame containing true labels, predictions, and metadata.
        val_slice_df (pd.DataFrame): DataFrame containing slice information for validation set.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate Per-Patient Loss
    # Weights: C1-C7 = 1.0, patient_overall = 7.0
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Extract arrays
    y_true = df_results[[f"{c}_true" for c in Config.TARGET_COLS]].values
    y_pred = df_results[[f"{c}_pred" for c in Config.TARGET_COLS]].values

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate Weighted BCE for each patient
    # Formula: -w * [y * log(p) + (1-y) * log(1-p)]
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    weighted_bce = bce * weights  # Shape: (N_patients, 8)

    # Average loss per patient (mean across the 8 classes)
    patient_losses = np.mean(weighted_bce, axis=1)
    df_results["patient_loss"] = patient_losses

    # 2. Correlate with Input Features
    # Feature: Number of slices per patient (scan depth)
    slice_counts = (
        val_slice_df.groupby("StudyInstanceUID").size().reset_index(name="num_slices")
    )

    # Merge loss with features
    df_analysis = pd.merge(df_results, slice_counts, on="StudyInstanceUID", how="left")

    # Calculate Correlation
    if len(df_analysis) > 1:
        # Correlation: Error Magnitude vs Number of Slices
        corr_slices = df_analysis["patient_loss"].corr(df_analysis["num_slices"])
        print(f"Correlation (Error Magnitude vs Input Slice Count): {corr_slices:.6f}")

        # Correlation: Error Magnitude vs Ground Truth (Fracture Presence)
        # This tells us if we error more on positive or negative cases
        corr_pos = df_analysis["patient_loss"].corr(df_analysis["patient_overall_true"])
        print(f"Correlation (Error Magnitude vs Fracture Presence): {corr_pos:.6f}")

        # Print stats for worst failures
        print("\nTop 3 Worst Predictions (Highest Loss):")
        worst_cases = df_analysis.sort_values("patient_loss", ascending=False).head(3)
        for _, row in worst_cases.iterrows():
            print(
                f"  UID: {row['StudyInstanceUID']} | Loss: {row['patient_loss']:.4f} | "
                f"True: {int(row['patient_overall_true'])} | Pred: {row['patient_overall_pred']:.4f} | "
                f"Slices: {row['num_slices']}"
            )
    else:
        print("Not enough validation data for correlation analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting Fracture Detection Pipeline...")

    # 2. Training
    # run_training handles data preparation, model initialization, and the training loop.
    # It saves the best model to ./working/idea_1/best_model.pth
    print("\n[Step 1/3] Training Model...")
    run_training()

    # 3. Validation & Metric Calculation
    print("\n[Step 2/3] Validating and Analyzing Failures...")

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = FractureClassifier(pretrained=False)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Model checkpoint not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare Validation Data
    val_slice_df = prepare_validation_slice_df(load_cached_data=True)
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    val_dataset = FractureSliceDataset(
        val_slice_df,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("val"),
        is_test=True,  # Set to True to get StudyInstanceUIDs in __getitem__
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    results = []
    with torch.no_grad():
        for images, uids, slice_nums in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.cpu().numpy()

            # Unpack batch
            for i in range(len(uids)):
                row = {"StudyInstanceUID": uids[i]}
                for idx, col in enumerate(Config.TARGET_COLS):
                    row[col] = preds[i][idx]
                results.append(row)

    if not results:
        print("Error: No validation predictions generated.")
        return

    # Aggregate: Max Pooling per Patient
    df_pred_raw = pd.DataFrame(results)
    df_pred_agg = (
        df_pred_raw.groupby("StudyInstanceUID")[Config.TARGET_COLS].max().reset_index()
    )

    # Merge with Ground Truth
    df_merge = pd.merge(
        val_metadata, df_pred_agg, on="StudyInstanceUID", suffixes=("_true", "_pred")
    )

    # Compute Final Metric
    y_true = df_merge[[f"{c}_true" for c in Config.TARGET_COLS]].values
    y_pred = df_merge[[f"{c}_pred" for c in Config.TARGET_COLS]].values

    final_metric = weighted_log_loss(y_true, y_pred)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric:.18f}")

    # Perform Failure Analysis
    perform_failure_analysis(df_merge, val_slice_df)

    # 4. Submission
    print("\n[Step 3/3] Generating Submission...")
    # run_inference handles loading test data, predicting, and saving submission.csv
    run_inference(load_cached_data=True)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
