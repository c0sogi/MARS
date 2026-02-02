import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from datetime import datetime
import warnings

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.data import load_and_preprocess_data, get_dataloaders, InsultDataset
from library.model import InsultModel
from library.engine import get_optimizer_params, train_fn, eval_fn

# Suppress warnings
warnings.filterwarnings("ignore")


def parse_date(date_str):
    """Parses date string to datetime object."""
    if pd.isna(date_str) or date_str == "":
        return pd.NaT
    try:
        clean_str = str(date_str).replace("Z", "")
        return datetime.strptime(clean_str, "%Y%m%d%H%M%S")
    except ValueError:
        return pd.NaT


def get_analysis_features(df):
    """Generates features for failure analysis."""
    df = df.copy()

    # Text features (Comment is already cleaned in the dataframe)
    df["char_len"] = df["Comment"].apply(lambda x: len(str(x)))
    df["word_len"] = df["Comment"].apply(lambda x: len(str(x).split()))

    # Date features
    dates = df["Date"].apply(parse_date)
    df["has_date"] = dates.notna().astype(int)
    df["hour"] = dates.apply(lambda x: x.hour if pd.notna(x) else -1)
    df["day_of_week"] = dates.apply(lambda x: x.weekday() if pd.notna(x) else -1)

    return df[["char_len", "word_len", "has_date", "hour", "day_of_week"]]


def run_training(train_df):
    """Runs Stratified K-Fold Training."""
    print(f"Starting training with {Config.n_folds} folds...")

    for fold in range(Config.n_folds):
        print(f"\n=== Fold {fold} ===")

        # DataLoaders
        train_loader, val_loader = get_dataloaders(train_df, fold)

        # Model
        model = InsultModel()
        model.to(Config.device)

        # Optimizer with LLRD
        optimizer_grouped_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.learning_rate,
            decoder_lr=Config.learning_rate * 5,
            weight_decay=Config.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        # Scheduler
        num_train_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.epochs):
            train_loss = train_fn(
                train_loader, model, optimizer, Config.device, scheduler
            )
            val_loss, val_preds, val_targets = eval_fn(val_loader, model, Config.device)
            val_auc = calculate_auc(val_targets, val_preds)

            print(
                f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()


def predict_with_ensemble(df, model_paths):
    """Generates averaged predictions using multiple models."""
    dataset = InsultDataset(df, inference_only=True)
    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    avg_preds = np.zeros(len(df))

    for path in model_paths:
        model = InsultModel()
        model.load_state_dict(torch.load(path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)
                outputs = model(input_ids, attention_mask)
                preds = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.extend(preds)

        avg_preds += np.array(fold_preds)

        del model
        torch.cuda.empty_cache()

    avg_preds /= len(model_paths)
    return avg_preds


def main():
    set_seed(Config.seed)

    # 1. Load Data
    # train_df_full contains combined train+val with fold info
    train_df_full, test_df = load_and_preprocess_data(load_cached_data=True)

    # 2. Identify Hold-out Validation Set
    df_train_orig = pd.read_csv(Config.train_path)
    split_idx = len(df_train_orig)
    val_holdout_df = train_df_full.iloc[split_idx:].reset_index(drop=True)

    print(f"Total Training Samples (CV): {len(train_df_full)}")
    print(f"Hold-out Validation Samples: {len(val_holdout_df)}")

    # 3. Train Models
    run_training(train_df_full)

    # 4. Evaluate on Hold-out Validation Set
    print("\nEvaluating Ensemble on Hold-out Validation Set...")
    model_paths = [
        os.path.join(Config.working_dir, f"model_fold_{i}.pth")
        for i in range(Config.n_folds)
    ]

    val_preds = predict_with_ensemble(val_holdout_df, model_paths)
    val_targets = val_holdout_df["Insult"].values

    final_auc = calculate_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_holdout_df["pred"] = val_preds
    val_holdout_df["error"] = np.abs(val_holdout_df["Insult"] - val_holdout_df["pred"])

    analysis_df = get_analysis_features(val_holdout_df)
    analysis_df["error"] = val_holdout_df["error"]

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.9508866995073892
    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )
        test_preds = predict_with_ensemble(test_df, model_paths)

        # Load sample submission to ensure correct format
        submission_path = os.path.join(Config.input_dir, "sample_submission_null.csv")
        submission = pd.read_csv(submission_path)
        submission["Insult"] = test_preds

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
