import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_datasets, WIVSDataset, get_transforms
from library.model import WIVSNet
from library.train import run_fold, predict_test_set


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Adjust Config for fast baseline execution
    Config.EPOCHS = 10  # Reduced from 15 to ensure completion within time limits

    seed_everything(Config.SEED)
    device = get_device()

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load datasets using cached data if available
    train_ds_raw, val_ds_raw, test_ds = get_datasets(load_cached_data=True)

    # Merge train and validation sets for Stratified K-Fold CV
    all_images = np.concatenate([train_ds_raw.images, val_ds_raw.images], axis=0)
    all_labels = np.concatenate([train_ds_raw.labels, val_ds_raw.labels], axis=0)
    all_ids = np.concatenate([train_ds_raw.ids, val_ds_raw.ids], axis=0)

    # ==========================================
    # 3. Cross-Validation Training
    # ==========================================
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(all_labels), dtype=np.float32)
    oof_targets = np.zeros(len(all_labels), dtype=np.float32)

    # To store feature stats for failure analysis later
    # We compute these during the loop to save memory/time
    feature_stats = {
        "mean_intensity": np.zeros(len(all_labels)),
        "std_intensity": np.zeros(len(all_labels)),
        "nonzero_ratio": np.zeros(len(all_labels)),
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        # Split Data
        X_train, y_train, ids_train = (
            all_images[train_idx],
            all_labels[train_idx],
            all_ids[train_idx],
        )
        X_val, y_val, ids_val = (
            all_images[val_idx],
            all_labels[val_idx],
            all_ids[val_idx],
        )

        # Create Datasets
        train_fold_ds = WIVSDataset(
            X_train, y_train, ids_train, transform=get_transforms("train")
        )
        val_fold_ds = WIVSDataset(
            X_val, y_val, ids_val, transform=get_transforms("valid")
        )

        # Train Fold (Uses library function)
        _ = run_fold(fold_idx, train_fold_ds, val_fold_ds, device)

        # ------------------------------------------
        # Inference on Validation Fold for OOF
        # ------------------------------------------
        # Load best model for this fold
        model_path = os.path.join(Config.MODEL_DIR, f"wivsnet_fold{fold_idx}.pth")
        model = WIVSNet(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        val_loader = DataLoader(
            val_fold_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        fold_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(probs)

        fold_preds = np.concatenate(fold_preds).flatten()

        # Store predictions and targets
        oof_preds[val_idx] = fold_preds
        oof_targets[val_idx] = y_val

        # Compute simple image statistics for failure analysis
        # X_val shape: (N, H, W, C)
        # We compute mean/std per sample across all spatial/channel dims
        feature_stats["mean_intensity"][val_idx] = X_val.mean(axis=(1, 2, 3))
        feature_stats["std_intensity"][val_idx] = X_val.std(axis=(1, 2, 3))
        feature_stats["nonzero_ratio"][val_idx] = (X_val > 0).mean(axis=(1, 2, 3))

        # Cleanup
        del model, train_fold_ds, val_fold_ds, val_loader
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Validation Metrics
    # ==========================================
    final_auc = roc_auc_score(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    errors = np.abs(oof_targets - oof_preds)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": feature_stats["mean_intensity"],
            "std_intensity": feature_stats["std_intensity"],
            "nonzero_ratio": feature_stats["nonzero_ratio"],
        }
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set (averaging all folds)
        test_preds = predict_test_set(test_ds, Config.NUM_FOLDS, device)

        submission_df = pd.DataFrame(
            {"BraTS21ID": test_ds.ids, "MGMT_value": test_preds}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
