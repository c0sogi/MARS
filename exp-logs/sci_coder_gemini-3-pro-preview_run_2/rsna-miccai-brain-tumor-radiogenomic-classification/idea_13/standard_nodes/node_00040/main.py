import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders, get_sorted_files
from library.model import AsymmetricEfficientNet
from library.train import run_training, predict_and_submit

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"
WORKING_DIR = "./working/idea_13"
SUBMISSION_FILE = "./submission/submission.csv"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
ROI_CACHE_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 5

# Submission Threshold
METRIC_THRESHOLD = 0.6254545454545455


def analyze_failures(val_df, val_preds, val_targets, input_dir, roi_cache_path):
    """
    Performs failure analysis by correlating error with metadata features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    analysis_df = val_df.copy()
    analysis_df["prediction"] = val_preds
    analysis_df["error"] = errors

    # Feature 1: FLAIR Slice Count (Volume Depth)
    # We count files in the FLAIR directory for each subject
    flair_counts = []
    for _, row in analysis_df.iterrows():
        flair_path = os.path.join(input_dir, row["path_FLAIR"])
        # Quick count of files
        try:
            # We use os.listdir directly for speed here, assuming standard structure
            if os.path.exists(flair_path):
                cnt = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
            else:
                cnt = 0
        except:
            cnt = 0
        flair_counts.append(cnt)

    analysis_df["flair_slice_count"] = flair_counts

    # Feature 2: Relative ROI Anchor (Tumor Location)
    # Load from cache
    roi_map = {}
    if os.path.exists(roi_cache_path):
        try:
            cache_df = pd.read_parquet(roi_cache_path)
            roi_map = pd.Series(
                cache_df.relative_anchor.values, index=cache_df.BraTS21ID
            ).to_dict()
        except:
            pass

    analysis_df["roi_anchor"] = analysis_df["BraTS21ID"].map(roi_map).fillna(0.5)

    # Calculate Correlations
    # 1. Error vs Slice Count
    if analysis_df["flair_slice_count"].std() > 0:
        corr_slices, _ = pearsonr(
            analysis_df["error"], analysis_df["flair_slice_count"]
        )
        print(f"Correlation (Error vs FLAIR Slice Count): {corr_slices:.4f}")
    else:
        print("Correlation (Error vs FLAIR Slice Count): N/A (Constant values)")

    # 2. Error vs ROI Anchor
    if analysis_df["roi_anchor"].std() > 0:
        corr_anchor, _ = pearsonr(analysis_df["error"], analysis_df["roi_anchor"])
        print(f"Correlation (Error vs ROI Anchor Position): {corr_anchor:.4f}")
    else:
        print("Correlation (Error vs ROI Anchor Position): N/A (Constant values)")

    # Identify worst failures
    print("\nTop 5 Worst Failures:")
    worst_cases = analysis_df.sort_values("error", ascending=False).head(5)
    print(
        worst_cases[
            ["BraTS21ID", "MGMT_value", "prediction", "error", "flair_slice_count"]
        ]
    )


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Execution Device: {device}")

    # 2. Training
    # We use the provided library function to run the training loop.
    # It handles loading data, training, validation, early stopping, and saving the best model.
    print("\n--- Starting Training ---")
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        save_dir=WORKING_DIR,
    )

    # 3. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")

    # Load Validation Data
    # We need the loader to get the exact tensors and targets
    _, val_loader, _ = get_dataloaders(
        train_metadata_path=os.path.join(METADATA_DIR, "train.csv"),
        val_metadata_path=os.path.join(METADATA_DIR, "val.csv"),
        test_metadata_path=os.path.join(METADATA_DIR, "test.csv"),
        input_dir=INPUT_DIR,
        batch_size=BATCH_SIZE,
        load_cached_data=True,
    )

    # Load Best Model
    model = AsymmetricEfficientNet(pretrained=False)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model not found. Using random weights.")

    model.to(device)
    model.eval()

    # Inference
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # targets are already on cpu in loop usually, but let's ensure consistency

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(targets.numpy())
            all_preds.extend(probs)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Calculate Metric
    try:
        final_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_auc = 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    # Load validation metadata dataframe to map back to IDs
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Ensure alignment (DataLoader preserves order if shuffle=False, which it is for val)
    if len(val_df) == len(all_preds):
        analyze_failures(val_df, all_preds, all_targets, INPUT_DIR, ROI_CACHE_PATH)
    else:
        print(
            "Warning: Mismatch between validation dataframe size and prediction count. Skipping detailed failure analysis."
        )

    # 5. Submission
    if final_auc > METRIC_THRESHOLD:
        print(
            f"\nValidation Metric ({final_auc}) > Threshold ({METRIC_THRESHOLD}). Generating submission..."
        )
        predict_and_submit(
            model_path=BEST_MODEL_PATH,
            output_file=SUBMISSION_FILE,
            batch_size=BATCH_SIZE,
        )
    else:
        print(
            f"\nValidation Metric ({final_auc}) <= Threshold ({METRIC_THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
