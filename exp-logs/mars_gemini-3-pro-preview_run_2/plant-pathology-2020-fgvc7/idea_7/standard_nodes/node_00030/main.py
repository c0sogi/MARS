import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.train import run_fold
from library.inference import generate_submission
from library.data import process_train_data, get_transforms, AppleDataset
from library.model import AppleClassifier
from library.utils import seed_everything

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# Adjust epochs to ensure completion within 2 hours while maintaining performance.
# 2 Models * 5 Folds * 8 Epochs = 80 Epochs total. ~25-30 mins runtime on A100.
Config.EPOCHS = 8
Config.BATCH_SIZE = 4
Config.ACCUMULATION_STEPS = 4


def get_oof_predictions(model_name, img_size, fold_idx, device):
    """
    Generates Out-Of-Fold (OOF) predictions for a specific model and fold.
    Replicates the validation data loading logic to ensure alignment with ground truth.
    """
    # 1. Recreate the validation split
    df = process_train_data(load_cached_data=True)
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    folds = list(skf.split(df, df["stratify_label"]))
    _, val_idx = folds[fold_idx]
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    # 2. Create Dataset and Loader
    # We use is_test=True to get image_ids returned, but we manually set 'val' transforms
    val_dataset = AppleDataset(
        val_df,
        img_size=img_size,
        transforms=get_transforms(img_size, mode="val"),
        is_test=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    safe_model_name = model_name.replace(".", "_")
    weights_path = os.path.join(
        Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold_idx}.pth"
    )

    model = AppleClassifier(model_name=model_name, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference with TTA (Horizontal Flip)
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in val_loader:
            images = images.to(device)

            # Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            all_preds.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)

    # Return DataFrame with binary predictions [rust, scab]
    res_df = pd.DataFrame(all_preds, columns=["pred_rust", "pred_scab"])
    res_df["image_id"] = all_ids

    return res_df


def analyze_failures(oof_df):
    """
    Performs failure analysis by correlating error magnitude with image metadata.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate Error Magnitude
    # We use the Mean Absolute Error across the 4 reconstructed classes
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    pred_cols = [f"pred_{c}" for c in target_cols]

    # Calculate mean error per row
    error_per_row = np.abs(oof_df[target_cols].values - oof_df[pred_cols].values).mean(
        axis=1
    )
    oof_df["error_magnitude"] = error_per_row

    # 2. Extract Metadata Features
    meta_stats = []

    for idx, row in oof_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            if not os.path.exists(full_path):
                continue

            size_bytes = os.path.getsize(full_path)
            img = cv2.imread(full_path)
            if img is None:
                continue

            h, w, c = img.shape
            mean_intensity = img.mean()

            meta_stats.append(
                {
                    "image_id": row["image_id"],
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h > 0 else 0,
                    "mean_intensity": mean_intensity,
                    "file_size_bytes": size_bytes,
                }
            )
        except Exception:
            continue

    meta_df = pd.DataFrame(meta_stats)

    # Merge metadata with error stats
    if not meta_df.empty:
        analysis_df = pd.merge(oof_df, meta_df, on="image_id")

        # 3. Calculate Correlations
        features = [
            "width",
            "height",
            "aspect_ratio",
            "mean_intensity",
            "file_size_bytes",
        ]
        correlations = {}

        for feat in features:
            if feat in analysis_df.columns:
                # Pearson correlation [0, 1] returns (correlation, p-value)
                corr = np.corrcoef(analysis_df["error_magnitude"], analysis_df[feat])[
                    0, 1
                ]
                correlations[feat] = corr

        print("Correlation between Error Magnitude and Input Features:")
        for feat, corr in correlations.items():
            print(f"  {feat}: {corr:.4f}")
    else:
        print("Could not compute metadata features for failure analysis.")


def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Define models to train (Heterogeneous Ensemble)
    model_configs = [
        (Config.MODEL_A_NAME, Config.IMG_SIZE_EFFNET),
        (Config.MODEL_B_NAME, Config.IMG_SIZE_CONVNEXT),
    ]

    # Storage for OOF predictions
    # Dictionary mapping image_id -> list of prediction arrays (one from each model)
    oof_preds_accumulator = {}

    # -------------------------------------------------------------------------
    # 1. Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training Pipeline...")
    for model_name, img_size in model_configs:
        for fold in range(Config.N_FOLDS):
            # Train the model for this fold
            run_fold(fold, model_name, img_size)

            # Generate OOF Preds immediately after training
            fold_oof_df = get_oof_predictions(model_name, img_size, fold, device)

            # Accumulate predictions
            for _, row in fold_oof_df.iterrows():
                img_id = row["image_id"]
                preds = np.array([row["pred_rust"], row["pred_scab"]])

                if img_id not in oof_preds_accumulator:
                    oof_preds_accumulator[img_id] = []
                oof_preds_accumulator[img_id].append(preds)

    # -------------------------------------------------------------------------
    # 2. Aggregation and Metric Calculation
    # -------------------------------------------------------------------------
    print("\nCalculating Final Validation Metric...")

    # Load Ground Truth
    gt_df = process_train_data(load_cached_data=True)

    final_preds = []

    for img_id, pred_list in oof_preds_accumulator.items():
        # Average predictions across all models (Ensemble)
        avg_preds = np.mean(pred_list, axis=0)
        p_r, p_s = avg_preds[0], avg_preds[1]

        # Reconstruct 4-class probabilities
        # Healthy: (1-r)(1-s)
        p_healthy = (1 - p_r) * (1 - p_s)
        # Multiple: r*s
        p_multiple = p_r * p_s
        # Rust Only: r(1-s)
        p_rust_only = p_r * (1 - p_s)
        # Scab Only: (1-r)s
        p_scab_only = (1 - p_r) * p_s

        final_preds.append(
            {
                "image_id": img_id,
                "pred_healthy": p_healthy,
                "pred_multiple_diseases": p_multiple,
                "pred_rust": p_rust_only,
                "pred_scab": p_scab_only,
            }
        )

    pred_df = pd.DataFrame(final_preds)

    # Merge with Ground Truth to ensure alignment
    val_merged = pd.merge(gt_df, pred_df, on="image_id")

    # Calculate Metric (Mean Column-wise ROC AUC)
    y_true = val_merged[["healthy", "multiple_diseases", "rust", "scab"]].values
    y_pred = val_merged[
        ["pred_healthy", "pred_multiple_diseases", "pred_rust", "pred_scab"]
    ].values

    val_auc = roc_auc_score(y_true, y_pred, average="macro")

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(val_merged)

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    # Strict threshold check
    THRESHOLD = 0.9954104122251848

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
