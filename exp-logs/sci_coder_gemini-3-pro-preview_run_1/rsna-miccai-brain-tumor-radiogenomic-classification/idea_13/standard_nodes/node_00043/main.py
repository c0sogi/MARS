import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.train import run_fold
from library.inference import predict_and_submit
from library.model import build_model
from library.data import get_dataloaders, get_data


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline execution
    Config.NUM_EPOCHS = 10

    logger = setup_logger("RunFile")
    set_seed(Config.SEED)

    logger.info("Initializing workflow...")

    # ==========================================
    # 2. Training
    # ==========================================
    logger.info("Starting Training Phase...")
    # run_fold handles the training loop and saves 'best_model.pth'
    # It returns the best instance-level AUC, but we need subject-level later.
    _ = run_fold(load_cached_data=True)

    # ==========================================
    # 3. Validation Assessment (Subject-Level)
    # ==========================================
    logger.info("Starting Validation Assessment...")

    device = torch.device(Config.DEVICE)

    # Load the best model trained in the previous step
    model = build_model(device)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        logger.error("Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Validation Data and Loader
    # get_data returns ((train), (val), (test))
    # We need val_ids to group predictions by subject
    _, (val_images, val_labels, val_ids), _ = get_data(load_cached_data=True)

    # get_dataloaders returns (train_loader, val_loader, test_loader, test_ids)
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Inference loop on Validation Set
    val_probs = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy().flatten())

    val_probs = np.concatenate(val_probs)

    # Create DataFrame for Aggregation
    # Note: val_ids, val_labels, and val_probs are aligned by index (instance level)
    df_val = pd.DataFrame(
        {"BraTS21ID": val_ids, "MGMT_value": val_labels, "prob": val_probs}
    )

    # Aggregate to Subject Level (Consensus Aggregation)
    # We take the mean of the probabilities for the 3 instances per subject
    df_subject = (
        df_val.groupby("BraTS21ID")
        .agg(
            {
                "MGMT_value": "mean",  # Should be constant for a subject (0 or 1)
                "prob": "mean",
            }
        )
        .reset_index()
    )

    # Calculate Final Metric
    final_metric = roc_auc_score(df_subject["MGMT_value"], df_subject["prob"])

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude
    df_subject["error"] = (df_subject["MGMT_value"] - df_subject["prob"]).abs()

    # Extract Metadata Features (File Counts) for Validation Subjects
    # We read the metadata file to get paths
    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    def get_file_count(rel_path):
        full_path = os.path.join("./input", rel_path)
        if os.path.exists(full_path):
            return len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
        return 0

    features = []
    for _, row in df_meta.iterrows():
        sid = row["BraTS21ID"]
        # Only process if subject is in our validation results
        if sid in df_subject["BraTS21ID"].values:
            features.append(
                {
                    "BraTS21ID": sid,
                    "flair_count": get_file_count(row["flair_path"]),
                    "t1w_count": get_file_count(row["t1w_path"]),
                    "t1wce_count": get_file_count(row["t1wce_path"]),
                    "t2w_count": get_file_count(row["t2w_path"]),
                }
            )

    df_features = pd.DataFrame(features)

    # Merge features with error data
    df_analysis = pd.merge(df_subject, df_features, on="BraTS21ID")

    # Calculate and Print Correlations
    feature_cols = ["flair_count", "t1w_count", "t1wce_count", "t2w_count"]
    print("Correlation between Error and Metadata Features:")
    for col in feature_cols:
        if col in df_analysis.columns:
            # Check if column has variance
            if df_analysis[col].std() > 0:
                corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
                print(f"{col}: {corr:.4f}")
            else:
                print(f"{col}: NaN (No variance)")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    threshold = 0.6705454545454544

    if final_metric > threshold:
        logger.info(
            f"Validation metric {final_metric} > {threshold}. Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        logger.info(
            f"Validation metric {final_metric} <= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
