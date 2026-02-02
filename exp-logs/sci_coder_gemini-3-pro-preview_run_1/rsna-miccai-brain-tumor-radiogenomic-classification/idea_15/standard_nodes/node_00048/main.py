import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# ==========================================
# 1. Configuration Patching (Fast Baseline)
# ==========================================
import library.config

# Override config for speed as per "Fast Baseline" requirement
library.config.NUM_EPOCHS = 5
library.config.NUM_FOLDS = 2

# Import library modules after patching so they pick up the new values
from library.config import (
    VAL_METADATA_PATH,
    TRAIN_METADATA_PATH,
    WORKING_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FOLDS,
    SEED,
    INPUT_DIR,
)
from library.utils import seed_everything
from library.dataset import SlabDataset, get_transforms
from library.model import WITSNetwork
from library.train import run_fold
from library.predict import inference_fn


def main():
    # Setup
    seed_everything(SEED)
    print(
        f"Configuration: NUM_EPOCHS={library.config.NUM_EPOCHS}, NUM_FOLDS={library.config.NUM_FOLDS}"
    )

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n=== Starting Training Phase ===")
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_METADATA_PATH}")

    train_meta = pd.read_csv(TRAIN_METADATA_PATH)

    # GroupKFold Split
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    groups = train_meta["BraTS21ID"]

    # Run training for the specified number of folds
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_meta, train_meta["MGMT_value"], groups)
    ):
        print(f"Running Fold {fold}...")
        run_fold(fold, train_idx, val_idx, train_meta)

    # ==========================================
    # 3. Validation Phase (Hold-out Set)
    # ==========================================
    print("\n=== Starting Validation Phase (Hold-out) ===")
    if not os.path.exists(VAL_METADATA_PATH):
        raise FileNotFoundError(f"Validation metadata not found at {VAL_METADATA_PATH}")

    val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Create Dataset for Hold-out Validation
    val_ds = SlabDataset(
        val_meta,
        transform=get_transforms("val"),
        load_cached_data=True,
        split_name="val_holdout",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Inference with Ensemble of Trained Models
    model = WITSNetwork().to(DEVICE)
    fold_preds_list = []

    models_found = 0
    for fold in range(NUM_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models_found += 1

            preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(DEVICE)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds.append(probs)

            # Concatenate batches and flatten
            fold_preds_list.append(np.concatenate(preds).flatten())
        else:
            print(f"Warning: Model for fold {fold} not found.")

    if models_found == 0:
        print("No models trained. Using random predictions for validation.")
        avg_slab_preds = np.random.rand(len(val_ds))
    else:
        # Average predictions across folds
        avg_slab_preds = np.mean(np.stack(fold_preds_list), axis=0)

    # Aggregate Slab Predictions to Subject Predictions
    # val_ds.ids contains the BraTS21ID for each slab
    df_pred = pd.DataFrame(
        {"BraTS21ID": val_ds.ids, "prob": avg_slab_preds, "target": val_ds.labels}
    )

    # Group by Subject ID (Mean of slabs)
    df_agg = (
        df_pred.groupby("BraTS21ID")
        .agg({"prob": "mean", "target": "first"})
        .reset_index()
    )

    # Compute Metric
    final_auc = roc_auc_score(df_agg["target"], df_agg["prob"])
    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate Error
    df_agg["error"] = (df_agg["target"] - df_agg["prob"]).abs()

    # Extract Metadata Features for Correlation (File Counts)
    print("Extracting file count features for correlation analysis...")
    feature_rows = []
    for _, row in val_meta.iterrows():
        sid = row["BraTS21ID"]
        feats = {"BraTS21ID": sid}
        for mod in ["flair", "t1w", "t1wce", "t2w"]:
            # Path relative to input dir
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                # Count dicom files
                count = len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
                feats[f"{mod}_count"] = count
            else:
                feats[f"{mod}_count"] = 0
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)

    # Merge with errors
    df_analysis = df_agg.merge(df_features, on="BraTS21ID")

    # Calculate Correlations
    print("Correlation between Error and File Counts:")
    feature_cols = [c for c in df_analysis.columns if c.endswith("_count")]
    for col in feature_cols:
        corr = df_analysis["error"].corr(df_analysis[col])
        print(f"  {col}: {corr:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.6705454545454544
    print(f"\n=== Submission Check ===")
    print(f"Threshold: {threshold}")
    print(f"Achieved:  {final_auc}")

    if final_auc > threshold:
        print("Metric check passed. Generating submission...")
        inference_fn(load_cached_data=True)
    else:
        print("Metric check failed. Skipping submission generation.")


if __name__ == "__main__":
    main()
