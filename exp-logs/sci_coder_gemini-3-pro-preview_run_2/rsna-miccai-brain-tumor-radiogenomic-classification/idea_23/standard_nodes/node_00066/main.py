import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_train_val_datasets, get_test_dataset
from library.model import AsymmetricEfficientNet
from library.train import run_training


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting pipeline execution...")
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Training Phase
    # --------------------------------------------------------------------------
    # Execute the training loop provided in the library.
    # This handles data loading, model initialization, training, and saving the best model.
    print("\n--- Phase 1: Training ---")
    run_training(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Phase 2: Validation Assessment ---")

    # Load the best model saved during training
    model = AsymmetricEfficientNet()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    _, val_dataset = get_train_val_datasets(load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference on Validation Set
    val_targets = []
    val_preds = []
    val_ids = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets.numpy().flatten())

            # Map predictions back to BraTS21IDs for analysis
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + inputs.size(0)
            # Access the underlying dataframe of the dataset
            batch_ids = val_dataset.df.iloc[start_idx:end_idx]["BraTS21ID"].values
            val_ids.extend(batch_ids)

    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)

    # Calculate and Print Final Metric
    try:
        final_metric = roc_auc_score(val_targets, val_preds)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Phase 3: Failure Analysis ---")
    errors = np.abs(val_targets - val_preds)

    # Construct DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": val_ids,
            "error": errors,
            "target": val_targets,
            "pred": val_preds,
        }
    )

    # Merge with metadata to get file paths for feature extraction
    val_meta_df = val_dataset.df
    df_analysis = df_analysis.merge(val_meta_df, on="BraTS21ID", how="left")

    # Extract simple meta-features (slice counts) to check for correlation with error
    print("Extracting metadata features for error correlation...")
    meta_features = []
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for idx, row in df_analysis.iterrows():
        feat = {}
        for mod in modalities:
            path_col = f"path_{mod}"
            full_path = os.path.join(Config.INPUT_DIR, row[path_col])
            if os.path.exists(full_path):
                # Count files in the directory
                count = len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
            else:
                count = 0
            feat[f"{mod}_count"] = count
        meta_features.append(feat)

    df_meta_feats = pd.DataFrame(meta_features)
    df_analysis = pd.concat([df_analysis, df_meta_feats], axis=1)

    # Calculate correlations
    print("Correlation between Prediction Error and Modality Slice Counts:")
    for col in df_meta_feats.columns:
        if df_analysis[col].std() > 0:
            corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: NaN (Constant)")

    # --------------------------------------------------------------------------
    # 4. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.6303636363636363

    if final_metric > threshold:
        print(f"\n--- Phase 4: Submission Generation ---")
        print(
            f"Validation metric ({final_metric}) > threshold ({threshold}). Generating predictions..."
        )

        test_dataset = get_test_dataset(load_cached_data=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)

                # Test-Time Augmentation (TTA)
                # 1. Original
                logits_orig = model(inputs)
                probs_orig = torch.sigmoid(logits_orig)

                # 2. Horizontal Flip (dim 3 is width in NCHW)
                inputs_hflip = torch.flip(inputs, [3])
                logits_hflip = model(inputs_hflip)
                probs_hflip = torch.sigmoid(logits_hflip)

                # 3. Vertical Flip (dim 2 is height in NCHW)
                inputs_vflip = torch.flip(inputs, [2])
                logits_vflip = model(inputs_vflip)
                probs_vflip = torch.sigmoid(logits_vflip)

                # Average probabilities
                avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

                test_preds.extend(avg_probs.cpu().numpy().flatten())

                # Get IDs
                start_idx = i * Config.BATCH_SIZE
                end_idx = start_idx + inputs.size(0)
                batch_ids = test_dataset.df.iloc[start_idx:end_idx]["BraTS21ID"].values
                test_ids.extend(batch_ids)

        # Create submission file
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\n--- Phase 4: Submission Skipped ---")
        print(f"Validation metric ({final_metric}) <= threshold ({threshold}).")


if __name__ == "__main__":
    main()
