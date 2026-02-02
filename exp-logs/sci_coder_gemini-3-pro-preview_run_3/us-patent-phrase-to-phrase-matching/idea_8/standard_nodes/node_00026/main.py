import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.dataset import load_dataset, create_folds, PhraseDataset
from library.model import CustomModel
from library.engine import train_fn, valid_fn, inference_fn, get_optimizer_params
from library.utils import seed_everything, get_score, AWP
from library.loss import HybridLoss


def run():
    # 1. Setup
    # Instantiate Config to ensure directories are created
    _ = Config()

    # Override Config for Fast Baseline Execution
    Config.epochs = 3  # Reduce epochs to ensure completion within 2 hours

    seed_everything(Config.seed)

    print(f"Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Model: {Config.model_name}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.batch_size}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 2. Data Loading
    print("\nLoading Datasets...")
    # Load metadata/train.csv for 5-Fold CV
    train_df = load_dataset("train")
    # Load metadata/val.csv for Hold-out Evaluation
    holdout_val_df = load_dataset("val")
    # Load metadata/test.csv for Inference
    test_df = load_dataset("test")

    print(f"  Train Shape: {train_df.shape}")
    print(f"  Hold-out Val Shape: {holdout_val_df.shape}")
    print(f"  Test Shape: {test_df.shape}")

    # Create Folds on the Training set
    train_df = create_folds(train_df, n_folds=Config.n_folds, seed=Config.seed)

    # 3. Training Loop (5-Fold CV)
    best_models_paths = []

    for fold in range(Config.n_folds):
        print(f"\n{'='*20} Fold {fold+1}/{Config.n_folds} {'='*20}")

        # Split Data
        fold_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val = train_df[train_df["fold"] == fold].reset_index(drop=True)

        # Create Datasets
        train_ds = PhraseDataset(fold_train, tokenizer, max_length=Config.max_length)
        valid_ds = PhraseDataset(fold_val, tokenizer, max_length=Config.max_length)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomModel(pretrained=True)
        model.to(Config.device)

        # Optimizer (AdamW with LLRD)
        optimizer_params = get_optimizer_params(
            model,
            encoder_lr=Config.encoder_lr,
            decoder_lr=Config.head_lr,
            weight_decay=Config.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            optimizer_params, eps=Config.eps, betas=Config.betas
        )

        # Scheduler (Cosine with Warmup)
        num_train_steps = int(len(fold_train) / Config.batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss, Scaler (Mixed Precision), and AWP
        criterion = HybridLoss()
        scaler = torch.cuda.amp.GradScaler()
        awp = (
            AWP(
                model,
                optimizer,
                adv_lr=Config.awp_lr,
                adv_eps=Config.awp_eps,
                start_epoch=Config.awp_start_epoch,
                scaler=scaler,
            )
            if Config.use_awp
            else None
        )

        # Training Loop
        best_score = -1
        best_model_path = os.path.join(
            Config.model_output_dir, f"model_fold_{fold}.pth"
        )

        for epoch in range(Config.epochs):
            # Train
            avg_train_loss = train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                Config.device,
                awp,
                scaler,
            )

            # Validate
            avg_val_loss, score = valid_fn(
                valid_loader, model, criterion, Config.device
            )

            print(
                f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Pearson: {score:.4f}"
            )

            # Save Best
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_model_path)
                print(f"  >>> New Best Score! Model Saved.")

        best_models_paths.append(best_model_path)

        # Cleanup to free memory
        del (
            model,
            optimizer,
            scheduler,
            scaler,
            awp,
            train_loader,
            valid_loader,
            train_ds,
            valid_ds,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Hold-out Validation (Ensemble)
    print(f"\n{'='*20} Hold-out Validation (Ensemble) {'='*20}")

    val_ds = PhraseDataset(holdout_val_df, tokenizer, max_length=Config.max_length)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    all_fold_preds = []

    for path in best_models_paths:
        print(f"Predicting with {os.path.basename(path)}...")
        model = CustomModel(pretrained=False)
        model.load_state_dict(torch.load(path, map_location=Config.device))
        model.to(Config.device)

        preds = inference_fn(val_loader, model, Config.device)
        all_fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions
    avg_preds = np.mean(all_fold_preds, axis=0)
    final_metric = get_score(holdout_val_df["score"].values, avg_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")
    holdout_val_df["pred"] = avg_preds
    holdout_val_df["error"] = (holdout_val_df["score"] - holdout_val_df["pred"]).abs()

    # Calculate features for correlation
    holdout_val_df["anchor_len"] = holdout_val_df["anchor"].astype(str).apply(len)
    holdout_val_df["target_len"] = holdout_val_df["target"].astype(str).apply(len)
    holdout_val_df["context_len"] = (
        holdout_val_df["context_text"].astype(str).apply(len)
    )

    # Calculate correlations
    features = ["error", "score", "anchor_len", "target_len", "context_len"]
    correlations = holdout_val_df[features].corr()["error"].drop("error")

    print("Correlation between Error and Input Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.8698034882545471

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) > Threshold ({THRESHOLD:.5f}). Generating Submission..."
        )

        test_ds = PhraseDataset(
            test_df, tokenizer, max_length=Config.max_length, inference_only=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds_list = []

        for path in best_models_paths:
            model = CustomModel(pretrained=False)
            model.load_state_dict(torch.load(path, map_location=Config.device))
            model.to(Config.device)

            preds = inference_fn(test_loader, model, Config.device)
            test_preds_list.append(preds)

            del model
            torch.cuda.empty_cache()
            gc.collect()

        final_test_preds = np.mean(test_preds_list, axis=0)

        submission = pd.DataFrame({"id": test_df["id"], "score": final_test_preds})

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nMetric ({final_metric:.5f}) <= Threshold ({THRESHOLD:.5f}). Submission Skipped."
        )


if __name__ == "__main__":
    run()
