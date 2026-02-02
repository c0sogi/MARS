import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config, utils, data_loader, model, train, inference


def main():
    # 1. Setup
    print("Initializing pipeline...")
    utils.seed_everything(config.SEED)

    # Override config for fast baseline execution
    config.EPOCHS = 5
    print(f"Configuration: EPOCHS={config.EPOCHS}, FOLDS={config.NUM_FOLDS}")

    # 2. Prepare Data & Metadata for OOF Tracking
    # We need to replicate the split logic to map predictions back to IDs for failure analysis
    print("Loading metadata for cross-validation tracking...")
    X_train_raw, y_train_raw, ids_train = data_loader.load_and_cache_data(
        config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )
    X_val_raw, y_val_raw, ids_val = data_loader.load_and_cache_data(
        config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    # Merge datasets as done in data_loader.get_dataloaders
    X_full = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_full = np.concatenate([y_train_raw, y_val_raw], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Storage for OOF results
    oof_ids = []
    oof_targets = []
    oof_probs = []

    # 3. Training Loop
    device = config.DEVICE
    criterion = nn.BCEWithLogitsLoss()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nProcessing Fold {fold_idx}/{config.NUM_FOLDS - 1}")

        # A. Train
        # train.run_fold handles data loading, training, and saving the best checkpoint
        train.run_fold(fold_idx)

        # B. Validation Inference (to get raw probabilities for OOF)
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load best model for this fold
        net = model.ACWIVNet(
            backbone_name=config.BACKBONE,
            pretrained=False,
            input_channels=config.INPUT_CHANNELS,
        )
        net = net.to(device)

        checkpoint_path = os.path.join(
            config.WORKING_DIR, f"best_model_fold{fold_idx}.pth"
        )
        utils.load_checkpoint(net, checkpoint_path, device=device)
        net.eval()

        # Get validation data for this fold
        # Note: We use the indices from our local skf split to ensure alignment with ids_full
        X_val_fold = X_full[val_idx]
        y_val_fold = y_full[val_idx]
        ids_val_fold = ids_full[val_idx]

        val_dataset = data_loader.VolumetricDataset(
            X_val_fold, y_val_fold, transforms=data_loader.get_transforms("val")
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        fold_probs = []
        fold_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = net(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

                fold_probs.extend(probs)
                fold_targets.extend(targets.numpy())

        # Store results
        oof_ids.extend(ids_val_fold)
        oof_targets.extend(fold_targets)
        oof_probs.extend(
            np.concatenate(fold_probs)
            if isinstance(fold_probs[0], np.ndarray)
            else fold_probs
        )

    # 4. Metric Calculation
    oof_targets = np.array(oof_targets)
    oof_probs = np.array(oof_probs)

    final_metric = roc_auc_score(oof_targets, oof_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error
    errors = np.abs(oof_targets - oof_probs)

    # Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": oof_ids,
            "Target": oof_targets,
            "Prob": oof_probs,
            "Error": errors,
        }
    )

    # Extract Feature: Scan Depth (File Count)
    # We read the metadata files again to get paths and count files
    print("Extracting metadata features (Scan Depth)...")
    df_meta_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_meta_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_meta_all = pd.concat([df_meta_train, df_meta_val], ignore_index=True)

    # Map ID to FLAIR file count
    id_to_count = {}
    for _, row in df_meta_all.iterrows():
        sid = row["BraTS21ID"]
        flair_path = os.path.join(config.INPUT_DIR, row["flair_path"])
        try:
            # Simple count of files in directory
            if os.path.exists(flair_path):
                count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
            else:
                count = 0
        except Exception:
            count = 0
        id_to_count[sid] = count

    df_analysis["Scan_Depth"] = df_analysis["BraTS21ID"].map(id_to_count)

    # Correlation
    if df_analysis["Scan_Depth"].std() > 0:
        corr, _ = pearsonr(df_analysis["Error"], df_analysis["Scan_Depth"])
        print(
            f"Correlation between Error and Scan Depth (FLAIR slice count): {corr:.4f}"
        )
    else:
        print("Scan Depth variance is 0, cannot compute correlation.")

    # 6. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        inference.predict(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
