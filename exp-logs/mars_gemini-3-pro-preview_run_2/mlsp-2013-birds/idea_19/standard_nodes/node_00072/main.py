import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.utils import set_seed, worker_init_fn, compute_metric
from library.dataset import BirdDataset
from library.models import get_model
from library.engine import train_one_epoch, validate, inference_fn
from skmultilearn.model_selection import IterativeStratifiedKFold
import torch.nn as nn


def load_tabular_features(filepath):
    """
    Loads the histogram of segments features for failure analysis.
    """
    if not os.path.exists(filepath):
        return None

    data_rows = []
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        start_idx = 0
        if "rec_id" in lines[0]:
            start_idx = 1

        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if len(parts) > 1:
                rec_id = int(parts[0])
                features = [float(x) for x in parts[1:]]
                data_rows.append([rec_id] + features)

        num_features = len(data_rows[0]) - 1
        cols = ["rec_id"] + [f"feat_{i}" for i in range(num_features)]
        return pd.DataFrame(data_rows, columns=cols)
    except Exception as e:
        print(f"Error loading tabular features: {e}")
        return None


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    N_FOLDS = 5

    # Architectures for the ensemble
    ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # 2. Data Preparation for CV
    print("Preparing Data for Cross-Validation...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    # Shuffle for reproducibility
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_all["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df_all.columns if c.startswith("species_")]
    y = df_all[label_cols].values

    # Test Dataset (loaded once)
    test_dataset = BirdDataset(
        csv_file=TEST_CSV,
        mode="test",
        load_cached_data=True,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
        height=224,
        width=448,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    kfold = IterativeStratifiedKFold(n_splits=N_FOLDS, order=1)

    # Storage for OOF and Test predictions
    oof_preds = np.zeros(y.shape)
    test_preds_sum = np.zeros((len(test_dataset), 19))

    # 3. CV Loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

        # Save temporary CSVs
        fold_train_df = df_all.iloc[train_idx]
        fold_val_df = df_all.iloc[val_idx]

        train_csv_path = os.path.join(WORKING_DIR, f"train_fold_{fold}.csv")
        val_csv_path = os.path.join(WORKING_DIR, f"val_fold_{fold}.csv")
        fold_train_df.to_csv(train_csv_path, index=False)
        fold_val_df.to_csv(val_csv_path, index=False)

        # Calculate pos_weight for BCE
        train_labels = fold_train_df[label_cols].values
        pos_counts = np.sum(train_labels, axis=0)
        neg_counts = len(train_labels) - pos_counts
        pos_counts = np.maximum(pos_counts, 1)
        pos_weights = neg_counts / pos_counts
        pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

        # Datasets & Loaders
        train_dataset = BirdDataset(
            csv_file=train_csv_path,
            mode="train",
            load_cached_data=True,
            cache_dir=os.path.join(WORKING_DIR, "cache"),
            height=224,
            width=448,
        )
        val_dataset = BirdDataset(
            csv_file=val_csv_path,
            mode="val",
            load_cached_data=True,
            cache_dir=os.path.join(WORKING_DIR, "cache"),
            height=224,
            width=448,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )

        for arch in ARCHITECTURES:
            print(f"  Training {arch}...")
            model = get_model(arch, num_classes=19, pretrained=True)
            model = model.to(device)

            # Use Weighted BCE Loss (Cite solution_lesson_node_00071)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

            best_val_loss = float("inf")
            best_model_path = os.path.join(WORKING_DIR, f"model_{arch}_fold_{fold}.pth")

            for epoch in range(NUM_EPOCHS):
                train_loss = train_one_epoch(
                    model, optimizer, None, train_loader, device, criterion
                )

                # Validation
                val_loss, val_auc = validate(model, val_loader, criterion, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)

            # Load best model
            model.load_state_dict(torch.load(best_model_path, map_location=device))

            # Predict OOF (with TTA)
            preds = inference_fn(model, val_loader, device)
            oof_preds[val_idx] += preds

            # Predict Test (with TTA)
            t_preds = inference_fn(model, test_loader, device)
            test_preds_sum += t_preds

    # Average OOF preds (divided by number of models)
    oof_preds /= len(ARCHITECTURES)

    # Average Test preds (divided by total models trained = folds * architectures)
    avg_test_preds = test_preds_sum / (N_FOLDS * len(ARCHITECTURES))

    # Compute Final Metric
    final_metric = compute_metric(y, oof_preds)
    print(f"\nFinal CV Validation Metric (OOF): {final_metric}")

    # Setup for failure analysis
    y_true = y
    ensemble_preds = oof_preds
    val_df_analysis = df_all.copy()

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    sample_errors = np.mean(np.abs(y_true - ensemble_preds), axis=1)

    # Load tabular features
    feat_path = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    df_feats = load_tabular_features(feat_path)

    if df_feats is not None:
        # Merge errors with features based on rec_id
        df_val = val_df_analysis.copy()
        df_val["error"] = sample_errors

        # Merge
        df_analysis = df_val[["rec_id", "error"]].merge(
            df_feats, on="rec_id", how="inner"
        )

        if len(df_analysis) > 0:
            correlations = []
            feat_cols = [c for c in df_analysis.columns if c.startswith("feat_")]

            for col in feat_cols:
                if df_analysis[col].std() > 0:  # Avoid constant columns
                    corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
                    correlations.append((col, corr))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Top 5 Feature Correlations with Error:")
            for name, corr in correlations[:5]:
                print(f"{name}: {corr:.4f}")
        else:
            print("No matching records for failure analysis.")
    else:
        print("Could not load tabular features for failure analysis.")

    # 6. Submission
    THRESHOLD = 0.9129501920716607

    if final_metric > THRESHOLD:
        print("\nMetric exceeds threshold. Generating submission...")

        # Format Submission
        # Format: Id, Probability
        # Id = rec_id * 100 + species_id

        submission_rows = []
        df_test = test_dataset.df

        for idx, row in df_test.iterrows():
            rec_id = int(row["rec_id"])
            probs = avg_test_preds[idx]

            for species_id, prob in enumerate(probs):
                sub_id = rec_id * 100 + species_id
                submission_rows.append([sub_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        # Sort by Id just in case
        df_sub = df_sub.sort_values("Id")

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
