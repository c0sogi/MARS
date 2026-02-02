import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import BirdDataset
from library.model import BirdClassifier
from library.engine import train_one_epoch, evaluate
from library.utils import seed_everything, calculate_roc_auc
from library.transforms import get_transforms


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Create directories for fold CSVs
    folds_dir = os.path.join(Config.WORKING_DIR, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    # Load Main Training Data
    if not os.path.exists(Config.TRAIN_CSV):
        print(f"Error: Training metadata not found at {Config.TRAIN_CSV}")
        return

    df_train_full = pd.read_csv(Config.TRAIN_CSV)

    # Prepare for Splitting
    # X is just a placeholder for the splitter, y is the targets
    X = df_train_full["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df_train_full.columns if c.startswith("species_")]
    y = df_train_full[label_cols].values

    # 2. Define Models and Training Loop
    # We will train 3 architectures * 5 folds = 15 models

    # Generate Folds using Iterative Stratification for multi-label balance
    # If the dataset is too small for strict stratification, this tries its best
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    fold_indices = []
    for train_idx, val_idx in k_fold.split(X, y):
        fold_indices.append((train_idx, val_idx))

    trained_models = []

    # Loop Architectures
    for arch_key, arch_conf in Config.MODEL_CONFIGS.items():
        model_name = arch_conf["model_name"]
        # Config defines img_size as (Freq, Time) -> (Height, Width)
        img_size = arch_conf["img_size"]
        target_h, target_w = img_size[0], img_size[1]

        print(
            f"Training Architecture: {model_name} | Resolution: {target_h}x{target_w}"
        )

        for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
            # Create temporary CSVs for this fold
            df_fold_train = df_train_full.iloc[train_idx].reset_index(drop=True)
            df_fold_val = df_train_full.iloc[val_idx].reset_index(drop=True)

            train_fold_csv = os.path.join(folds_dir, f"train_fold_{fold_i}.csv")
            val_fold_csv = os.path.join(folds_dir, f"val_fold_{fold_i}.csv")

            df_fold_train.to_csv(train_fold_csv, index=False)
            df_fold_val.to_csv(val_fold_csv, index=False)

            # Dataset & Dataloader
            train_transforms = get_transforms("train", width=target_w, height=target_h)
            val_transforms = get_transforms("val", width=target_w, height=target_h)

            train_dataset = BirdDataset(
                train_fold_csv, transform=train_transforms, preload=True
            )
            val_dataset = BirdDataset(
                val_fold_csv, transform=val_transforms, preload=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model Setup
            model = BirdClassifier(model_name=model_name, pretrained=True)
            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Calculate Pos Weight for this fold to handle class imbalance
            if Config.USE_POS_WEIGHT:
                train_labels = df_fold_train[label_cols].values
                num_pos = np.sum(train_labels, axis=0)
                num_neg = len(train_labels) - num_pos
                num_pos = np.maximum(num_pos, 1)  # Avoid division by zero
                pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(
                    device
                )
            else:
                pos_weight = None

            # Training Loop
            best_auc = 0.0
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold{fold_i}.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, optimizer, train_loader, device, pos_weight
                )

                # Check performance on fold-val set to determine best model
                val_loss, val_preds, val_targets = evaluate(model, val_loader, device)
                auc = calculate_roc_auc(val_targets, val_preds)

                if auc > best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), best_model_path)

            # Keep track of trained model paths
            trained_models.append(
                {
                    "path": best_model_path,
                    "model_name": model_name,
                    "height": target_h,
                    "width": target_w,
                }
            )

            # Cleanup to save memory
            del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache()

    # 3. Final Validation on Hold-Out Set
    print("Running Final Validation on Hold-Out Set...")

    if not os.path.exists(Config.VAL_CSV):
        print(f"Error: Validation metadata not found at {Config.VAL_CSV}")
        return

    df_val_holdout = pd.read_csv(Config.VAL_CSV)
    y_true_holdout = df_val_holdout[label_cols].values

    # Initialize aggregate array
    agg_preds = np.zeros_like(y_true_holdout, dtype=np.float64)

    for model_info in trained_models:
        # Load Dataset for this resolution
        h, w = model_info["height"], model_info["width"]
        transforms = get_transforms("val", width=w, height=h)
        ds = BirdDataset(Config.VAL_CSV, transform=transforms, preload=True)
        dl = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Load Model
        model = BirdClassifier(model_name=model_info["model_name"], pretrained=False)
        model.load_state_dict(torch.load(model_info["path"], map_location=device))
        model.to(device)
        model.eval()

        _, preds, _ = evaluate(model, dl, device)
        agg_preds += preds

        del model, ds, dl
        torch.cuda.empty_cache()

    # Average predictions (Ensemble)
    avg_preds = agg_preds / len(trained_models)

    # Metric
    final_auc = calculate_roc_auc(y_true_holdout, avg_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    # Error per sample (Mean Absolute Error across labels)
    sample_mae = np.mean(np.abs(y_true_holdout - avg_preds), axis=1)

    # Feature: Number of species (Cardinality)
    sample_cardinality = np.sum(y_true_holdout, axis=1)

    # Correlation
    if np.std(sample_mae) > 0 and np.std(sample_cardinality) > 0:
        corr, _ = pearsonr(sample_mae, sample_cardinality)
        print(f"Correlation between Error and Label Cardinality: {corr:.4f}")
    else:
        print("Correlation could not be calculated (zero variance).")

    # 5. Submission
    threshold = 0.9129501920716607
    if final_auc > threshold:
        print("Generating Submission...")

        if not os.path.exists(Config.TEST_CSV):
            print(f"Error: Test metadata not found at {Config.TEST_CSV}")
            return

        df_test = pd.read_csv(Config.TEST_CSV)
        test_agg_preds = np.zeros((len(df_test), Config.NUM_CLASSES), dtype=np.float64)

        for model_info in trained_models:
            h, w = model_info["height"], model_info["width"]
            transforms = get_transforms("test", width=w, height=h)
            ds = BirdDataset(Config.TEST_CSV, transform=transforms, preload=True)
            dl = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            model = BirdClassifier(
                model_name=model_info["model_name"], pretrained=False
            )
            model.load_state_dict(torch.load(model_info["path"], map_location=device))
            model.to(device)
            model.eval()

            _, preds, _ = evaluate(model, dl, device)
            test_agg_preds += preds

            del model, ds, dl
            torch.cuda.empty_cache()

        test_avg_preds = test_agg_preds / len(trained_models)

        # Flatten for submission: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = test_avg_preds[i]
            for species_idx, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        sub_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Validation metric {final_auc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
