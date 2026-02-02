import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from transformers import AutoTokenizer, logging as hf_logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_pearson
from library.dataset import load_and_preprocess_data, PhraseDataset
from library.model import PhraseSimilarityModel
from library.engine import train_fn, eval_fn, inference_fn

# Suppress warnings and logs
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)


def main():
    # 1. Setup and Configuration
    seed_everything(Config.seed)

    # Override Config for fast baseline execution
    Config.epochs = 5
    Config.n_folds = 4
    Config.train_batch_size = 16
    Config.valid_batch_size = 32
    Config.model_output_dir = os.path.join(Config.output_dir, "models")
    os.makedirs(Config.model_output_dir, exist_ok=True)

    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    # Load metadata splits
    df_train_full = load_and_preprocess_data("train", load_cached_data=True)
    df_val_holdout = load_and_preprocess_data("val", load_cached_data=True)
    df_test = load_and_preprocess_data("test", load_cached_data=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Stratified Group K-Fold Training
    # We split df_train_full into K folds
    skf = StratifiedGroupKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Stratification targets and groups
    # Note: score is float, we map to string for stratification
    y_strata = df_train_full["score"].astype(str)
    groups = df_train_full["anchor"]

    trained_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, y_strata, groups=groups)
    ):
        print(f"\n--- Fold {fold + 1}/{Config.n_folds} ---")

        # Prepare Fold Data
        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

        train_dataset = PhraseDataset(df_train_fold, tokenizer, mode="train")
        val_dataset = PhraseDataset(df_val_fold, tokenizer, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
        )

        # Model Setup
        model = PhraseSimilarityModel(pretrained=True)
        model.to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        num_training_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Training Loop
        best_pearson = -1.0
        best_model_path = os.path.join(
            Config.model_output_dir, f"model_fold_{fold}.bin"
        )

        for epoch in range(Config.epochs):
            train_loss, train_pearson = train_fn(
                model, train_loader, optimizer, scheduler, device, scaler, epoch
            )
            val_loss, val_pearson = eval_fn(model, val_loader, device)

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Pearson: {val_pearson:.4f}"
            )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                torch.save(model.state_dict(), best_model_path)

        trained_model_paths.append(best_model_path)

        # Cleanup
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Final Validation on Hold-out Set
    print("\nRunning Final Validation on Hold-out Set...")

    val_dataset = PhraseDataset(df_val_holdout, tokenizer, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    # Ensemble Prediction
    fold_preds = []
    for path in trained_model_paths:
        model = PhraseSimilarityModel(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.to(device)

        preds = inference_fn(model, val_loader, device)
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    avg_val_preds = np.mean(fold_preds, axis=0)

    # Compute Metric
    final_metric = compute_pearson(avg_val_preds, df_val_holdout["score"].values)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    df_val_holdout["pred"] = avg_val_preds
    df_val_holdout["error"] = np.abs(df_val_holdout["score"] - df_val_holdout["pred"])

    # Feature extraction for analysis
    df_val_holdout["len_anchor"] = df_val_holdout["anchor"].astype(str).apply(len)
    df_val_holdout["len_target"] = df_val_holdout["target"].astype(str).apply(len)

    # Correlations
    corr_len_anchor, _ = pearsonr(df_val_holdout["error"], df_val_holdout["len_anchor"])
    corr_len_target, _ = pearsonr(df_val_holdout["error"], df_val_holdout["len_target"])
    corr_score_mag, _ = pearsonr(df_val_holdout["error"], df_val_holdout["score"])

    print(f"Correlation (Error vs Anchor Length): {corr_len_anchor:.4f}")
    print(f"Correlation (Error vs Target Length): {corr_len_target:.4f}")
    print(f"Correlation (Error vs Ground Truth Score): {corr_score_mag:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.8043391108512878

    if final_metric > THRESHOLD:
        print("\nMetric passed threshold. Generating submission...")

        test_dataset = PhraseDataset(df_test, tokenizer, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
        )

        test_fold_preds = []
        for path in trained_model_paths:
            model = PhraseSimilarityModel(pretrained=False)
            model.load_state_dict(torch.load(path))
            model.to(device)

            preds = inference_fn(model, test_loader, device)
            test_fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_fold_preds, axis=0)

        # Clip
        avg_test_preds = np.clip(avg_test_preds, 0.0, 1.0)

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        submission_df = pd.DataFrame({"id": df_test["id"], "score": avg_test_preds})

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
