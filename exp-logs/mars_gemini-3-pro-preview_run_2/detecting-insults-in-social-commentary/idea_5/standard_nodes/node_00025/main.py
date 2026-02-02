import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import warnings
import copy

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import (
    load_train_data,
    load_test_data,
    prepare_loaders,
    prepare_test_loader,
    merge_pseudo_labels,
    InsultDataset,
    get_folds,
)
from library.model import InsultModel
from library.engine import get_optimizer_params, train_fn, valid_fn
from library.awp import AWP

# Suppress warnings
warnings.filterwarnings("ignore")


def run_training():
    # Setup
    seed_everything(Config.seed)
    os.makedirs(Config.output_dir, exist_ok=True)
    device = Config.device

    print(f"Device: {device}")
    print("Loading Data...")

    # Load initial data
    train_df = load_train_data()
    test_df = load_test_data()

    # Placeholders for OOF and Test predictions
    oof_preds_stage1 = np.zeros(len(train_df))
    test_preds_stage1 = []

    # ====================================================
    # Stage 1: Teacher Training
    # ====================================================
    print("\n" + "=" * 30)
    print("Stage 1: Teacher Training")
    print("=" * 30)

    # We use the fold definition from library to ensure consistency
    df_folds = get_folds(train_df, Config.n_folds, Config.seed)

    for fold in range(Config.n_folds):
        print(f"\nFold {fold+1}/{Config.n_folds}")

        # Prepare loaders
        train_loader, valid_loader = prepare_loaders(fold, df=train_df)

        # Model
        model = InsultModel(pretrained=True)
        model.to(device)

        # Optimizer
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.lr,
            decoder_lr=Config.lr,
            weight_decay=Config.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=Config.lr, eps=1e-6, betas=(0.9, 0.999)
        )

        # Scheduler
        num_train_steps = int(len(train_loader) * Config.stage1_epochs)
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Criterion
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_score = -np.inf
        best_model_wts = None

        for epoch in range(Config.stage1_epochs):
            avg_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                awp=None,
            )

            score, preds, val_loss = valid_fn(valid_loader, model, criterion, device)
            print(f"Epoch {epoch+1} - val_loss: {val_loss:.4f} - val_auc: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model_wts = copy.deepcopy(model.state_dict())

        # Load best model for inference
        model.load_state_dict(best_model_wts)

        # Store OOF predictions
        # Identify indices for this fold
        val_idx = df_folds[df_folds["fold"] == fold].index

        # Re-run validation to get predictions exactly matching best weights
        score, val_preds, _ = valid_fn(valid_loader, model, criterion, device)
        oof_preds_stage1[val_idx] = val_preds

        # Predict on Test
        test_loader = prepare_test_loader()
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                y_preds = model(**inputs)
                fold_test_preds.append(y_preds.sigmoid().cpu().numpy())
        test_preds_stage1.append(np.concatenate(fold_test_preds))

        # Clean up
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()

    # Calculate Stage 1 Score
    stage1_auc = get_score(train_df["Insult"].values, oof_preds_stage1)
    print(f"Stage 1 CV AUC: {stage1_auc:.6f}")

    # ====================================================
    # Pseudo-Labeling
    # ====================================================
    print("\n" + "=" * 30)
    print("Generating Pseudo-Labels")
    print("=" * 30)

    avg_test_preds = np.mean(test_preds_stage1, axis=0)
    augmented_df = merge_pseudo_labels(
        train_df, test_df, avg_test_preds, threshold=Config.pseudo_label_threshold
    )
    print(f"Original Train Size: {len(train_df)}")
    print(f"Augmented Train Size: {len(augmented_df)}")

    # ====================================================
    # Stage 2: Student Training (AWP)
    # ====================================================
    print("\n" + "=" * 30)
    print("Stage 2: Student Training with AWP")
    print("=" * 30)

    oof_preds_stage2 = np.zeros(len(train_df))
    test_preds_stage2 = []

    # Pre-load tokenizer for custom dataset creation
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    for fold in range(Config.n_folds):
        print(f"\nFold {fold+1}/{Config.n_folds}")

        # 1. Identify Validation Indices (Original Data Only)
        val_idx = df_folds[df_folds["fold"] == fold].index
        valid_df = train_df.loc[val_idx].reset_index(drop=True)

        # 2. Identify Training Indices (Original Data)
        train_idx = df_folds[df_folds["fold"] != fold].index
        train_df_fold = train_df.loc[train_idx]

        # 3. Get Pseudo-Labeled Data
        # The pseudo-labels are appended at the end of augmented_df
        pseudo_part = augmented_df.iloc[len(train_df) :]

        # 4. Combine for Training
        train_df_final = pd.concat([train_df_fold, pseudo_part], ignore_index=True)

        # Create Datasets & Loaders
        train_dataset = InsultDataset(train_df_final, tokenizer, Config.max_len)
        valid_dataset = InsultDataset(valid_df, tokenizer, Config.max_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model Setup
        model = InsultModel(pretrained=True)
        model.to(device)

        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.lr,
            decoder_lr=Config.lr,
            weight_decay=Config.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=Config.lr, eps=1e-6)

        num_train_steps = int(len(train_loader) * Config.stage2_epochs)
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        criterion = nn.BCEWithLogitsLoss()

        # AWP Setup
        awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

        best_score = -np.inf
        best_model_wts = None

        for epoch in range(Config.stage2_epochs):
            # Enable AWP only after start epoch
            use_awp_now = (
                awp if (epoch >= Config.awp_start_epoch and Config.use_awp) else None
            )

            avg_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                awp=use_awp_now,
            )

            score, preds, val_loss = valid_fn(valid_loader, model, criterion, device)
            print(f"Epoch {epoch+1} - val_loss: {val_loss:.4f} - val_auc: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model_wts = copy.deepcopy(model.state_dict())

        # Load best
        model.load_state_dict(best_model_wts)

        # OOF Preds
        score, val_preds, _ = valid_fn(valid_loader, model, criterion, device)
        oof_preds_stage2[val_idx] = val_preds

        # Test Preds
        test_loader = prepare_test_loader()
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                y_preds = model(**inputs)
                fold_test_preds.append(y_preds.sigmoid().cpu().numpy())
        test_preds_stage2.append(np.concatenate(fold_test_preds))

        del model, optimizer, scheduler, train_loader, valid_loader, awp
        torch.cuda.empty_cache()

    # ====================================================
    # Evaluation & Submission
    # ====================================================
    print("\n" + "=" * 30)
    print("Final Evaluation")
    print("=" * 30)

    final_cv_score = get_score(train_df["Insult"].values, oof_preds_stage2)
    print(f"Final Validation Metric: {final_cv_score}")

    # Failure Analysis
    print("\nFailure Analysis:")
    targets = train_df["Insult"].values
    errors = np.abs(targets - oof_preds_stage2)

    # Features for correlation
    lengths = train_df["Comment"].apply(lambda x: len(str(x))).values
    word_counts = train_df["Comment"].apply(lambda x: len(str(x).split())).values

    corr_len = np.corrcoef(errors, lengths)[0, 1]
    corr_word = np.corrcoef(errors, word_counts)[0, 1]

    print(f"Correlation (Error vs Char Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Word Count): {corr_word:.4f}")

    # Submission
    threshold = 0.9632101806239738
    if final_cv_score > threshold:
        print(
            f"\nMetric ({final_cv_score}) > Threshold ({threshold}). Generating Submission..."
        )

        avg_test_preds_stage2 = np.mean(test_preds_stage2, axis=0)

        sample_sub_path = os.path.join(Config.input_dir, "sample_submission_null.csv")
        if os.path.exists(sample_sub_path):
            sub_df = pd.read_csv(sample_sub_path)
            if "Unnamed: 0" in sub_df.columns:
                sub_df = sub_df.drop(columns=["Unnamed: 0"])

            sub_df["Insult"] = avg_test_preds_stage2
            sub_df.to_csv(Config.submission_path, index=False)
            print(f"Submission saved to {Config.submission_path}")
        else:
            print("Sample submission file not found. Creating minimal submission.")
            sub_df = pd.DataFrame({"Insult": avg_test_preds_stage2})
            sub_df.to_csv(Config.submission_path, index=False)
    else:
        print(
            f"\nMetric ({final_cv_score}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run_training()
