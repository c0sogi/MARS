import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Import from library
from library.config import CFG
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import BirdDataset, load_or_compute_spectrograms
from library.model import BirdResNet
from library.trainer import train_one_epoch, valid_one_epoch


def run():
    # 1. Setup
    # Override epochs for a fast baseline execution as requested
    CFG.epochs = 30

    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    # Ensure output directories exist
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Combine train and val for Stratified Cross Validation to maximize data usage
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Development samples: {len(dev_df)}")
    print(f"Test samples: {len(test_df)}")

    # 3. Load/Compute Spectrograms
    # Pass all dataframes to ensure all necessary files are processed and cached
    all_dfs = [dev_df, test_df]
    spec_cache = load_or_compute_spectrograms(all_dfs, load_cached_data=True)

    # 4. Prepare Test Loader
    test_dataset = BirdDataset(test_df, spec_cache, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Arrays to store results
    # OOF (Out of Fold) predictions for the entire dev set
    oof_preds = np.zeros((len(dev_df), CFG.num_classes))
    oof_targets = np.zeros((len(dev_df), CFG.num_classes))
    # Accumulator for test set predictions (Ensemble)
    test_preds_accum = np.zeros((len(test_df), CFG.num_classes))

    # Store indices to map OOF preds back correctly
    dev_indices = np.arange(len(dev_df))

    # 5. Cross Validation Loop
    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dev_df)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        fold_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        # Store targets for OOF evaluation
        # We need to extract the targets from the dataframe manually or via dataset
        # Using dataset logic to parse string labels to one-hot
        val_dataset_temp = BirdDataset(fold_val_df, spec_cache, phase="val")
        # Collect targets
        fold_val_targets = []
        for i in range(len(val_dataset_temp)):
            _, label, _ = val_dataset_temp[i]
            fold_val_targets.append(label.numpy())
        oof_targets[val_idx] = np.array(fold_val_targets)

        # Datasets & Loaders
        train_dataset = BirdDataset(fold_train_df, spec_cache, phase="train")
        val_dataset = BirdDataset(fold_val_df, spec_cache, phase="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Model, Criterion, Optimizer
        model = BirdResNet(pretrained=CFG.pretrained, num_classes=CFG.num_classes)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        # Training Loop
        best_score = -np.inf
        best_model_path = os.path.join(CFG.output_dir, f"fold_{fold}_best.pth")

        for epoch in range(CFG.epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
            val_loss, val_score, _ = valid_one_epoch(
                model, val_loader, criterion, device
            )

            # Save Best
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best AUC: {best_score:.10f}")

        # Reload Best Model for Inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 1. Predict on Validation Set (OOF)
        with torch.no_grad():
            fold_val_preds = []
            for images, _, _ in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                fold_val_preds.append(probs.cpu().numpy())
            fold_val_preds = np.concatenate(fold_val_preds)
            oof_preds[val_idx] = fold_val_preds

        # 2. Predict on Test Set (Accumulate)
        with torch.no_grad():
            fold_test_preds = []
            for images, _, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                fold_test_preds.append(probs.cpu().numpy())
            fold_test_preds = np.concatenate(fold_test_preds)
            test_preds_accum += fold_test_preds

    # 6. Calculate Final Metrics
    final_metric = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate Mean Absolute Error per sample
    sample_errors = np.abs(oof_targets - oof_preds).mean(axis=1)

    # Extract Features for correlation
    signal_means = []
    signal_stds = []
    label_counts = oof_targets.sum(axis=1)

    for rid in dev_df["rec_id"]:
        if rid in spec_cache:
            spec = spec_cache[rid]
            signal_means.append(np.mean(spec))
            signal_stds.append(np.std(spec))
        else:
            signal_means.append(0)
            signal_stds.append(0)

    analysis_df = pd.DataFrame(
        {
            "error": sample_errors,
            "signal_mean": signal_means,
            "signal_std": signal_stds,
            "label_count": label_counts,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 8. Submission
    threshold = 0.9072993371210134
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Average predictions
        avg_test_preds = test_preds_accum / CFG.n_folds

        submission_rows = []
        for i, row in test_df.iterrows():
            rec_id = int(row["rec_id"])
            probs = avg_test_preds[i]

            for species_idx in range(CFG.num_classes):
                # ID format: rec_id * 100 + species_number
                row_id = rec_id * 100 + species_idx
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df = submission_df.sort_values("Id")

        sub_path = os.path.join(CFG.submission_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
