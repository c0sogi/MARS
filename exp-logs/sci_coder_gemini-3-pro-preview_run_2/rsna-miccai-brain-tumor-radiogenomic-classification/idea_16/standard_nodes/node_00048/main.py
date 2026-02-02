import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import (
    VAL_METADATA_PATH,
    BEST_MODEL_PATH,
    CACHE_FILE_PATH,
    DEVICE,
    NUM_EPOCHS,
    INPUT_DIR,
    SEED,
)
from library.utils import seed_everything
from library.data import get_dataloader
from library.model import get_model
from library.train import run_training
from library.predict import predict_and_submit


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Training
    # Run training using the configuration defaults (20 epochs).
    # debug=False ensures we use the full dataset for the best possible performance.
    run_training(num_epochs=NUM_EPOCHS, debug=False)

    # 3. Validation & Metric Calculation
    if not os.path.exists(BEST_MODEL_PATH):
        print("Error: Best model file not found.")
        return

    # Load validation metadata
    if not os.path.exists(VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {VAL_METADATA_PATH}")
        return

    df_val = pd.read_csv(VAL_METADATA_PATH)

    # Initialize DataLoader for validation
    val_loader = get_dataloader(df_val, phase="val", shuffle=False)

    # Load the best model
    model = get_model()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Run Inference
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for inputs, targets, ids in val_loader:
            inputs = inputs.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    # Compute AUC
    if len(np.unique(all_targets)) < 2:
        val_auc = 0.5
    else:
        val_auc = roc_auc_score(all_targets, all_preds)

    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    # Construct analysis dataframe
    df_analysis = pd.DataFrame(
        {"BraTS21ID": all_ids, "target": all_targets, "pred": all_preds}
    )

    # Calculate error magnitude
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["pred"])

    # Ensure ID types match for merging
    df_val["BraTS21ID"] = df_val["BraTS21ID"].astype(int)
    df_analysis["BraTS21ID"] = df_analysis["BraTS21ID"].astype(int)

    # Merge with metadata
    df_merged = df_val.merge(df_analysis, on="BraTS21ID", how="inner")

    # Merge with ROI cache to get anchor_idx feature
    if os.path.exists(CACHE_FILE_PATH):
        try:
            roi_cache = pd.read_parquet(CACHE_FILE_PATH)
            roi_cache["BraTS21ID"] = roi_cache["BraTS21ID"].astype(int)
            df_merged = df_merged.merge(roi_cache, on="BraTS21ID", how="left")
            df_merged["anchor_idx"] = df_merged["anchor_idx"].fillna(0)
        except Exception:
            df_merged["anchor_idx"] = 0
    else:
        df_merged["anchor_idx"] = 0

    # Extract structural feature: Slice Count (proxy for brain volume/scan depth)
    def get_slice_count(rel_path):
        try:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                return len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
        except Exception:
            pass
        return 0

    df_merged["FLAIR_slices"] = df_merged["path_FLAIR"].apply(get_slice_count)

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    features_to_check = ["target", "anchor_idx", "FLAIR_slices"]

    for feat in features_to_check:
        if feat in df_merged.columns:
            # Calculate correlation if variance exists
            if df_merged[feat].std() > 0 and df_merged["error"].std() > 0:
                corr, _ = pearsonr(df_merged["error"], df_merged[feat])
                print(f"Feature: {feat}, Correlation: {corr:.4f}")
            else:
                print(f"Feature: {feat}, Correlation: NaN (Insufficient variance)")

    # 5. Submission
    SUBMISSION_THRESHOLD = 0.6254545454545455

    if val_auc > SUBMISSION_THRESHOLD:
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"Validation metric {val_auc} <= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
