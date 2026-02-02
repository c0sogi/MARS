import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloader
from library.model_arch import AsymmetricEfficientNet
from library.engine import train_model, evaluate, predict_consensus


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    # 1. Collect Predictions
    with torch.no_grad():
        for v1, v2, labels, ids in val_loader:
            v1 = v1.to(device)
            v2 = v2.to(device)

            # Forward both views
            logits1 = model(v1)
            logits2 = model(v2)

            # Consensus Probability
            prob1 = torch.sigmoid(logits1)
            prob2 = torch.sigmoid(logits2)
            avg_prob = (prob1 + prob2) / 2.0

            all_preds.extend(avg_prob.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    # 2. Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {"BraTS21ID": all_ids, "label": all_labels, "pred": all_preds}
    )

    # Calculate Error Magnitude
    df_analysis["error"] = (df_analysis["label"] - df_analysis["pred"]).abs()

    # 3. Load Metadata and ROI Features
    # Load base metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found. Skipping detailed feature correlation.")
        return

    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Load cached ROI indices (features)
    roi_cache_path = os.path.join(Config.WORKING_DIR, "roi_indices_val.parquet")
    if os.path.exists(roi_cache_path):
        df_roi = pd.read_parquet(roi_cache_path)
        # Merge features into analysis dataframe
        df_analysis = df_analysis.merge(df_roi, on="BraTS21ID", how="left")
    else:
        print("ROI cache not found. Skipping ROI correlation analysis.")
        df_roi = None

    # 4. Calculate Correlations
    print("Correlation between Error Magnitude and Features:")

    features_to_check = []
    if df_roi is not None:
        features_to_check.extend(["roi_anchor1_idx", "roi_anchor2_idx"])

    # Also check correlation with the label itself (class bias)
    features_to_check.append("label")

    for feature in features_to_check:
        if feature in df_analysis.columns:
            # Drop NaNs just in case
            valid_data = df_analysis.dropna(subset=[feature, "error"])
            if len(valid_data) > 1 and valid_data[feature].std() > 0:
                corr, _ = pearsonr(valid_data[feature], valid_data["error"])
                print(f"  Feature '{feature}': {corr:.4f}")
            else:
                print(f"  Feature '{feature}': N/A (Constant or insufficient data)")

    # Top failures
    print("\nTop 5 Worst Predictions:")
    print(
        df_analysis.sort_values("error", ascending=False).head(5)[
            ["BraTS21ID", "label", "pred", "error"]
        ]
    )


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to use pre-processed .npy files if available
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
    )
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )
    test_loader = get_dataloader(
        "test", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = AsymmetricEfficientNet().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    print("Starting Training...")
    # We use the config epochs, but the engine handles early stopping.
    model = train_model(
        model, train_loader, val_loader, optimizer, device, epochs=Config.EPOCHS
    )

    # 5. Validation Assessment
    print("Performing Final Validation...")
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.6254545454545455

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc:.6f}) > Threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        predict_consensus(model, test_loader, device)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({val_auc:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
