import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
    IMG_SIZE,
    MODALITIES,
    RELATIVE_DEPTHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    INPUT_DIR,
)
from library.utils import seed_everything, AverageMeter
from library.dataset import get_dataloader, BraTSDataset, get_transforms
from library.model import SIRVEfficientNet

# ====================================================
# Configuration for Fast Baseline
# ====================================================
N_FOLDS = 5
EPOCHS_PER_FOLD = 8  # Reduced slightly to ensure 2h runtime safety with 5 folds
THRESHOLD_AUC = 0.6705454545454544


def train_fold(fold_idx, train_df, val_df_fold):
    """
    Trains a single fold and returns the best model state dict.
    """
    print(f"\n--- Starting Fold {fold_idx} ---")

    # Create DataLoaders
    train_loader = get_dataloader(train_df, phase="train", batch_size=BATCH_SIZE)
    valid_loader = get_dataloader(val_df_fold, phase="valid", batch_size=BATCH_SIZE)

    # Initialize Model
    model = SIRVEfficientNet(pretrained=True)
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS_PER_FOLD + 1):
        # Training
        model.train()
        train_loss = AverageMeter()
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))

        # Validation (In-Fold)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        try:
            fold_auc = roc_auc_score(
                np.concatenate(all_targets), np.concatenate(all_preds)
            )
        except:
            fold_auc = 0.5

        # Save Best
        if fold_auc > best_auc:
            best_auc = fold_auc
            best_model_state = model.state_dict()

        # Silent progress to keep logs clean as requested
        # print(f"Fold {fold_idx} Ep {epoch}: AUC {fold_auc:.4f}")

    print(f"Fold {fold_idx} Best AUC: {best_auc:.4f}")
    return best_model_state


def run_inference(model_paths, df, phase="valid"):
    """
    Runs ensemble inference using saved models.
    """
    loader = get_dataloader(df, phase=phase, batch_size=BATCH_SIZE, shuffle=False)

    # We will store sum of probabilities and average later
    avg_preds = np.zeros(len(df))
    ids_list = []
    targets_list = []

    # Collect IDs and Targets once
    # Note: Loader order is deterministic if shuffle=False and workers=0 or consistent
    # We'll iterate the loader once to get ground truth/IDs, then iterate models

    # To be safe and efficient, we iterate models and pass data
    # But to avoid reloading data 5 times, let's load model, predict, unload.

    # Pre-allocate array based on loader size is tricky if drop_last, but phase!=train so no drop_last

    # Strategy: Iterate dataloader, for each batch, run all 5 models (if they fit in memory)
    # OR: Run model 1 on full dataset, then model 2... sum up.

    # Let's do: Run model 1 on full dataset, store preds. Repeat.

    final_preds = np.zeros(len(df))

    for model_path in model_paths:
        model = SIRVEfficientNet(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        fold_preds = []
        current_ids = []
        current_targets = []

        with torch.no_grad():
            for batch in loader:
                if phase == "test":
                    images, batch_ids = batch
                    images = images.to(DEVICE)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.extend(probs)
                    current_ids.extend(batch_ids)
                else:
                    images, targets = batch
                    images = images.to(DEVICE)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.extend(probs)
                    current_targets.extend(targets.numpy())

        final_preds += np.array(fold_preds)

        # Store targets/ids from the first pass
        if len(ids_list) == 0 and phase == "test":
            ids_list = current_ids
        if len(targets_list) == 0 and phase != "test":
            targets_list = current_targets

        del model
        torch.cuda.empty_cache()

    final_preds /= len(model_paths)

    return final_preds, ids_list, targets_list


def perform_failure_analysis(df, preds, targets):
    """
    Correlates error with a simple metadata feature (FLAIR file count).
    """
    print("\n--- Failure Analysis ---")
    df = df.copy()
    df["pred"] = preds
    df["target"] = targets
    df["error"] = (df["target"] - df["pred"]).abs()

    # Extract a feature: Number of FLAIR files
    # This is a proxy for scan resolution/depth
    flair_counts = []
    for _, row in df.iterrows():
        path = os.path.join(INPUT_DIR, row["flair_path"])
        try:
            count = len(os.listdir(path))
        except:
            count = 0
        flair_counts.append(count)

    df["flair_count"] = flair_counts

    # Correlation
    corr = df["error"].corr(df["flair_count"])
    print(f"Correlation between Error Magnitude and FLAIR Slice Count: {corr:.6f}")

    # Additional check: Error by class
    err_0 = df[df["target"] == 0]["error"].mean()
    err_1 = df[df["target"] == 1]["error"].mean()
    print(f"Mean Error Class 0: {err_0:.4f}")
    print(f"Mean Error Class 1: {err_1:.4f}")


def main():
    seed_everything(SEED)

    # 1. Load Metadata
    df_train_full = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_holdout = pd.read_csv(VAL_METADATA_PATH)

    # 2. 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_paths = []

    # We split the 'df_train_full' for CV training
    X = df_train_full
    y = df_train_full["MGMT_value"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_sub = X.iloc[train_idx].reset_index(drop=True)
        val_sub = X.iloc[val_idx].reset_index(drop=True)

        best_state = train_fold(fold, train_sub, val_sub)

        save_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
        torch.save(best_state, save_path)
        model_paths.append(save_path)

    # 3. Validation on Hold-out Set
    print("\n--- Validation on Hold-out Set ---")
    val_preds, _, val_targets = run_inference(
        model_paths, df_val_holdout, phase="valid"
    )

    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(df_val_holdout, val_preds, val_targets)

    # 5. Submission
    if final_auc > THRESHOLD_AUC:
        print("\n--- Generating Submission ---")
        df_test = pd.read_csv(TEST_METADATA_PATH)
        test_preds, test_ids, _ = run_inference(model_paths, df_test, phase="test")

        submission = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        # Ensure IDs are integers as per sample submission
        submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)

        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({THRESHOLD_AUC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
