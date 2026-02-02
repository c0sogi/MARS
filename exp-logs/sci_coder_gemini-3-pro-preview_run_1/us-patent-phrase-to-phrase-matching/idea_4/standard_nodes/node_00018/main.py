import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_pearson_score
from library.cpc_loader import get_cpc_texts
from library.dataset import load_processed_data, PhraseDataset
from library.model import PhraseModel, get_optimizer_grouped_parameters
from library.engine import train_fn, eval_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(df, preds, target_col="score"):
    """
    Analyzes model errors on the validation set.
    Computes correlation between absolute error and input features.
    """
    print("\n[Failure Analysis]")
    df_ana = df.copy()
    df_ana["pred"] = preds
    df_ana["error"] = (df_ana[target_col] - df_ana["pred"]).abs()

    # Create simple features for correlation analysis
    df_ana["anchor_len"] = df_ana["anchor"].astype(str).apply(len)
    df_ana["target_len"] = df_ana["target"].astype(str).apply(len)
    df_ana["context_len"] = df_ana["context_text"].astype(str).apply(len)

    # Select features to correlate with error
    features = ["anchor_len", "target_len", "context_len", target_col]
    correlations = df_ana[features].corrwith(df_ana["error"])

    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    seed_everything(Config.seed)
    Config.create_output_dir()
    device = Config.device

    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading CPC descriptions...")
    cpc_texts = get_cpc_texts(load_cached_data=True)

    print("Loading Train/Val Data...")
    # Load the official provided splits from metadata
    # We treat 'train_path' as the development set for Cross-Validation
    # We treat 'val_path' as the final hold-out set for reporting the metric
    train_full = load_processed_data(
        Config.train_path, cpc_texts, load_cached_data=True
    )
    val_holdout = load_processed_data(Config.val_path, cpc_texts, load_cached_data=True)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 4. Stratified Group K-Fold Setup
    # We use StratifiedGroupKFold to prevent anchor leakage between train/val splits within the CV loop.
    sgkf = StratifiedGroupKFold(n_splits=Config.n_folds)

    groups = train_full[Config.group_col]
    # Use string representation of scores for stratification bins
    y_bins = train_full[Config.target_col].astype(str)

    # FAST BASELINE: We only run Fold 0 to ensure completion within the time limit.
    target_fold = 0

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(train_full, y_bins, groups=groups)
    ):
        if fold_idx != target_fold:
            continue

        print(f"\n{'='*20} Training Fold {fold_idx} {'='*20}")

        # Create Fold DataFrames
        df_train = train_full.iloc[train_idx].reset_index(drop=True)
        df_val = train_full.iloc[val_idx].reset_index(drop=True)

        # Override epochs for fast baseline execution
        current_epochs = 2

        # Datasets
        train_dataset = PhraseDataset(df_train, tokenizer, max_length=Config.max_length)
        val_dataset = PhraseDataset(df_val, tokenizer, max_length=Config.max_length)

        # Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model
        model = PhraseModel(Config.model_name, pretrained=True)
        model.to(device)

        # Optimizer with Layer-wise Learning Rate Decay (LLRD)
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, Config.learning_rate, Config.weight_decay, Config.layer_decay
        )
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=Config.learning_rate,
            eps=Config.eps,
            betas=Config.betas,
        )

        # Scheduler
        num_train_steps = len(train_loader) * current_epochs
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=Config.num_cycles,
        )

        # Mixed Precision Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Training Loop
        best_score = -1.0
        best_model_path = os.path.join(Config.output_dir, f"model_fold_{fold_idx}.pth")

        for epoch in range(current_epochs):
            # Train (includes AWP logic internally)
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch, scheduler, device, scaler
            )

            # Eval on Fold Validation
            val_loss, val_score = eval_fn(val_loader, model, device)

            print(
                f"Epoch {epoch+1}/{current_epochs} - Train Loss: {avg_loss:.4f} - Val Pearson: {val_score:.4f}"
            )

            # Save Best
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved Best Model (Score: {best_score:.4f})")

        # Load Best Model for this fold
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # ---------------------------------------------------------
        # 5. Validation on Official Hold-out Set
        # ---------------------------------------------------------
        print(f"\nEvaluating Fold {fold_idx} model on Hold-out Validation Set...")

        holdout_dataset = PhraseDataset(
            val_holdout, tokenizer, max_length=Config.max_length
        )
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Inference
        holdout_preds = inference_fn(holdout_loader, model, device)
        holdout_labels = val_holdout[Config.target_col].values

        final_metric = compute_pearson_score(holdout_labels, holdout_preds)
        print(f"Final Validation Metric: {final_metric}")

        # 6. Failure Analysis
        run_failure_analysis(val_holdout, holdout_preds)

        # 7. Submission
        # Threshold check
        if final_metric > 0.8673:
            print("\nMetric > 0.8673. Generating Submission...")

            # Load Test Data
            test_df = load_processed_data(
                Config.test_path, cpc_texts, load_cached_data=True
            )
            test_dataset = PhraseDataset(
                test_df, tokenizer, max_length=Config.max_length
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.valid_batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Inference
            test_preds = inference_fn(test_loader, model, device)

            # Create Submission DataFrame
            submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})

            # Ensure output directory
            os.makedirs("./submission", exist_ok=True)
            submission_path = "./submission/submission.csv"
            submission.to_csv(submission_path, index=False)
            print(f"Submission saved to {submission_path}")

        else:
            print(f"\nMetric {final_metric} <= 0.8673. Submission skipped.")

        # Break after 1 fold for fast baseline
        break


if __name__ == "__main__":
    main()
