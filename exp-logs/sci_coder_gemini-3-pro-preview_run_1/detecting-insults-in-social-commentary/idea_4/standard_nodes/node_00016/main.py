import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_datasets
from library.model import HybridDeberta
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating error magnitude with features.
    """
    df = val_df.copy()
    df["pred"] = y_pred
    df["label"] = y_true
    df["error"] = np.abs(df["label"] - df["pred"])

    # Feature Engineering for Analysis
    df["char_count"] = df["Comment"].fillna("").apply(len)
    df["word_count"] = df["Comment"].fillna("").apply(lambda x: len(str(x).split()))

    def get_caps_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)

    df["caps_ratio"] = df["Comment"].fillna("").apply(get_caps_ratio)
    df["exclam_count"] = df["Comment"].fillna("").apply(lambda x: str(x).count("!"))

    features = ["char_count", "word_count", "caps_ratio", "exclam_count"]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print("-" * 40)
    for feat in features:
        corr = df[feat].corr(df["error"])
        print(f"{feat}: {corr:.4f}")
    print("-" * 40)


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Load Data
    # Load datasets using the library function
    # We use load_cached_data=True to use precomputed SVD features if available
    train_dataset_full, val_dataset_holdout, test_dataset, tokenizer = get_datasets(
        load_cached_data=True
    )

    # Load DataFrames for metadata/analysis purposes
    df_train_full = pd.read_csv(Config.train_path)
    df_val_holdout = pd.read_csv(Config.val_path)

    # 3. Stratified K-Fold Cross Validation
    # We split the 'train_dataset_full' into K folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Extract labels for stratification
    train_labels = df_train_full["Insult"].values

    # Store predictions
    # We need predictions on the holdout validation set from each fold model
    oof_val_preds = np.zeros((Config.n_folds, len(val_dataset_holdout)))
    test_preds = np.zeros((Config.n_folds, len(test_dataset)))

    # Prepare Holdout Validation Loader (Fixed across folds)
    val_loader_holdout = DataLoader(
        val_dataset_holdout,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Prepare Test Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"\nStarting {Config.n_folds}-Fold Cross-Validation...")

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(np.zeros(len(train_labels)), train_labels)
    ):
        print(f"\n=== Fold {fold + 1}/{Config.n_folds} ===")

        # Create Subsets
        train_subset = Subset(train_dataset_full, train_idx)
        valid_subset = Subset(train_dataset_full, valid_idx)

        # Create DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_subset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Trainer
        trainer = Trainer(train_loader, valid_loader, device=device)

        # Train
        # This saves 'best_model.bin' in working_dir
        best_model_path, best_auc = trainer.train()

        # Rename model for this fold to avoid overwriting
        fold_model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")
        shutil.move(best_model_path, fold_model_path)
        print(f"Fold {fold+1} Best Internal AUC: {best_auc:.4f}")

        # Predict on Holdout Validation Set
        print(f"Predicting on Holdout Validation Set (Fold {fold+1})...")
        fold_val_preds = trainer.predict(val_loader_holdout, fold_model_path)
        oof_val_preds[fold] = fold_val_preds

        # Predict on Test Set
        print(f"Predicting on Test Set (Fold {fold+1})...")
        fold_test_preds = trainer.predict(test_loader, fold_model_path)
        test_preds[fold] = fold_test_preds

        # Cleanup
        del trainer, train_loader, valid_loader, train_subset, valid_subset
        torch.cuda.empty_cache()

    # 4. Ensemble Predictions
    print("\nComputing Ensemble Predictions...")
    # Average predictions across folds
    avg_val_preds = np.mean(oof_val_preds, axis=0)
    avg_test_preds = np.mean(test_preds, axis=0)

    # 5. Final Evaluation
    y_true_val = df_val_holdout["Insult"].values
    final_val_auc = roc_auc_score(y_true_val, avg_val_preds)

    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    analyze_failures(df_val_holdout, y_true_val, avg_val_preds)

    # 7. Submission
    threshold = 0.9582101806239737

    if final_val_auc > threshold:
        print(
            f"Validation AUC ({final_val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Load test metadata to preserve structure
        df_test = pd.read_csv(Config.test_path)

        # Create submission DataFrame
        submission_df = df_test.copy()
        submission_df["Insult"] = avg_test_preds

        # Reorder columns to match sample: Insult, Date, Comment
        cols = ["Insult", "Date", "Comment"]
        # Ensure columns exist
        if "Date" not in submission_df.columns:
            submission_df["Date"] = ""

        submission_df = submission_df[cols]

        # Save
        os.makedirs(Config.submission_dir, exist_ok=True)
        submission_df.to_csv(Config.submission_file, index=False)
        print(f"Submission saved to {Config.submission_file}")

    else:
        print(
            f"Validation AUC ({final_val_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
