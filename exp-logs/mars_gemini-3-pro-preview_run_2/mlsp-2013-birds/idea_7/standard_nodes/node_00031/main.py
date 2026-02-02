import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import pearsonr

# Import provided library modules
from library.utils import set_seed, calculate_roc_auc
from library.dataset import BirdDataset, get_transforms
from library.models import BirdClassifier
from library.engine import train_one_epoch, validate

# --- Constants ---
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 25  # Sufficient for small dataset fine-tuning
LEARNING_RATE = 1e-4
NUM_CLASSES = 19
THRESHOLD_METRIC = 0.8739452549958209
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"


def load_tabular_features(rec_ids):
    """
    Loads the histogram of segments features for failure analysis.
    Returns a DataFrame aligned with the requested rec_ids.
    """
    hist_path = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    if not os.path.exists(hist_path):
        return None

    try:
        # Read lines to handle potential header issues manually
        with open(hist_path, "r") as f:
            lines = f.readlines()

        data = []
        start_idx = 0
        if "rec_id" in lines[0]:
            start_idx = 1

        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if len(parts) > 1:
                rid = int(parts[0])
                feats = [float(x) for x in parts[1:]]
                data.append([rid] + feats)

        cols = ["rec_id"] + [f"feat_{i}" for i in range(len(data[0]) - 1)]
        df_feats = pd.DataFrame(data, columns=cols)

        # Filter and align
        df_feats = df_feats[df_feats["rec_id"].isin(rec_ids)].set_index("rec_id")
        df_feats = df_feats.reindex(rec_ids).fillna(0)  # Fill missing with 0
        return df_feats

    except Exception as e:
        print(f"Warning: Could not load tabular features: {e}")
        return None


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Load Data
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Combine for CV
    df_dev = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Prepare X and y for stratification
    # X is dummy, we just need indices
    X_dummy = df_dev[["rec_id"]].values
    label_cols = [c for c in df_dev.columns if c.startswith("species_")]
    y_dev = df_dev[label_cols].values

    # 3. Cross-Validation Loop
    stratifier = IterativeStratification(
        n_splits=NUM_FOLDS,
        order=1,
        sample_distribution_per_fold=[1.0 / NUM_FOLDS] * NUM_FOLDS,
    )

    # Storage for OOF predictions and models
    oof_preds = np.zeros((len(df_dev), NUM_CLASSES))
    oof_targets = np.zeros((len(df_dev), NUM_CLASSES))

    # We will save model paths to load later for ensemble
    model_paths = []

    print(f"Starting {NUM_FOLDS}-Fold Cross-Validation...")

    fold_idx = 0
    for train_idx, val_idx in stratifier.split(X_dummy, y_dev):
        print(f"\n--- Fold {fold_idx} ---")

        df_fold_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_dev.iloc[val_idx].reset_index(drop=True)

        # Calculate pos_weight for this fold
        y_train_fold = df_fold_train[label_cols].values
        pos_counts = np.sum(y_train_fold, axis=0)
        neg_counts = len(df_fold_train) - pos_counts
        # Avoid division by zero
        pos_counts = np.where(pos_counts == 0, 1, pos_counts)
        pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(
            device
        )

        # Datasets & Loaders
        train_ds = BirdDataset(
            df_fold_train, phase="train", transform=get_transforms("train")
        )
        val_ds = BirdDataset(df_fold_val, phase="val", transform=get_transforms("val"))

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # We train two architectures per fold
        architectures = ["resnet18", "densenet121"]

        fold_preds = np.zeros((len(df_fold_val), NUM_CLASSES))

        for arch in architectures:
            # print(f"Training {arch}...")
            model = BirdClassifier(backbone=arch, num_classes=NUM_CLASSES).to(device)
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            best_auc = 0.0
            best_model_path = os.path.join(
                WORKING_DIR, f"model_fold{fold_idx}_{arch}.pth"
            )

            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, pos_weight
                )
                val_loss, val_auc = validate(model, val_loader, device, pos_weight)
                scheduler.step()

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # Load best model for inference on validation set
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            model_paths.append((arch, best_model_path))

            # Predict on val set
            arch_preds = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                    arch_preds.append(probs.cpu().numpy())

            fold_preds += np.concatenate(arch_preds, axis=0)

        # Average predictions from both architectures
        fold_preds /= len(architectures)

        # Store OOF
        oof_preds[val_idx] = fold_preds
        oof_targets[val_idx] = df_fold_val[label_cols].values

        fold_idx += 1

    # 4. Global Validation Metric
    final_auc = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    per_sample_mae = np.mean(np.abs(oof_targets - oof_preds), axis=1)

    # Load features
    df_features = load_tabular_features(df_dev["rec_id"].values)

    if df_features is not None:
        # Calculate correlations
        correlations = []
        for col in df_features.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(df_features[col]):
                # Align data
                feat_vals = df_features[col].values
                if len(np.unique(feat_vals)) > 1:  # Skip constant features
                    corr, _ = pearsonr(per_sample_mae, feat_vals)
                    if not np.isnan(corr):
                        correlations.append((col, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Feature Correlations with Error Magnitude:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")
    else:
        print("Skipping detailed feature correlation (tabular data missing).")

    # 6. Submission
    if final_auc > THRESHOLD_METRIC:
        print("\nMetric threshold met. Generating submission...")

        test_ds = BirdDataset(df_test, phase="test", transform=get_transforms("val"))
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize aggregate predictions
        test_preds_sum = np.zeros((len(df_test), NUM_CLASSES))

        # Iterate over all saved models
        for arch, path in model_paths:
            model = BirdClassifier(backbone=arch, num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()

            model_preds = []
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                    model_preds.append(probs.cpu().numpy())

            test_preds_sum += np.concatenate(model_preds, axis=0)

        # Average
        avg_test_preds = test_preds_sum / len(model_paths)

        # Format Submission
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = avg_test_preds[i]
            for species_idx, prob in enumerate(probs):
                # Id format: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_auc} did not meet threshold {THRESHOLD_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
