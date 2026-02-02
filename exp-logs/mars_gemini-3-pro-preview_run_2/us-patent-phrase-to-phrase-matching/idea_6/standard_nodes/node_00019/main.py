import os
import gc
import sys
import time
import math
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, get_score
from library.dataset import PhraseDataset
from library.model import HybridDeberta
from library.awp import AWP
from library.engine import train_fn, valid_fn
from library.feature_engineering import get_features_batch


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    CFG.setup()

    # Override epochs for speed constraints (Fast Baseline)
    # Reducing to 3 epochs ensures completion within the 2-hour limit on the A100
    CFG.epochs = 3
    print(f"Setting epochs to {CFG.epochs} for runtime constraint.")

    device = CFG.device
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # 2. Load Data
    print("Loading data...")
    train_df_full = pd.read_csv(CFG.train_file)
    holdout_val_df = pd.read_csv(CFG.val_file)
    test_df = pd.read_csv(CFG.test_file)

    # 3. Stratified K-Fold on train_df_full
    # We use the provided training data for CV to train the ensemble
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    # Prepare storage for OOF and Test predictions
    # We will store predictions for the holdout set from each fold to ensemble them
    holdout_preds_folds = []
    test_preds_folds = []

    # Create structural features for holdout and test once to ensure cache is ready
    print("Pre-computing features for holdout and test...")
    _ = get_features_batch(
        holdout_val_df["anchor"], holdout_val_df["target"], cache_name="val"
    )
    _ = get_features_batch(test_df["anchor"], test_df["target"], cache_name="test")

    # 4. Training Loop
    # Create bin for stratification to ensure balanced folds
    train_df_full["score_bin"] = pd.cut(train_df_full["score"], bins=5, labels=False)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df_full, train_df_full["score_bin"])
    ):
        print(f"\n{'='*20} Fold {fold+1} / {CFG.n_fold} {'='*20}")

        # Split data
        fold_train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

        # Datasets
        # Use unique cache names for training folds to avoid conflicts
        train_dataset = PhraseDataset(
            fold_train_df, tokenizer, mode="train", cache_name=f"train_fold{fold}"
        )
        valid_dataset = PhraseDataset(
            fold_val_df, tokenizer, mode="val", cache_name=f"val_fold{fold}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Model Initialization
        model = HybridDeberta(pretrained=True)
        model.to(device)

        # Optimizer parameters with differential learning rates
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": CFG.encoder_lr,
                "weight_decay": CFG.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": CFG.encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": CFG.decoder_lr,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_parameters, eps=CFG.eps, betas=CFG.betas
        )

        # Scheduler
        num_train_steps = int(
            len(fold_train_df)
            / CFG.train_batch_size
            / CFG.gradient_accumulation_steps
            * CFG.epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Initialize Adversarial Weight Perturbation (AWP)
        awp = AWP(
            model,
            optimizer,
            adv_lr=CFG.awp_lr,
            adv_eps=CFG.awp_eps,
            start_epoch=CFG.awp_start_epoch,
        )

        # Training Loop
        best_score = -1

        for epoch in range(CFG.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch, scheduler, device, awp
            )
            val_score, val_loss, _ = valid_fn(valid_loader, model, device)

            print(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {val_loss:.4f}  val_score: {val_score:.4f}"
            )

            if val_score > best_score:
                best_score = val_score
                torch.save(
                    model.state_dict(),
                    os.path.join(CFG.output_dir, f"model_fold_{fold}.bin"),
                )
                print(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f}")

        # Load best model for this fold
        model.load_state_dict(
            torch.load(os.path.join(CFG.output_dir, f"model_fold_{fold}.bin"))
        )
        model.eval()

        # Inference on Holdout Validation Set
        holdout_dataset = PhraseDataset(
            holdout_val_df, tokenizer, mode="val", cache_name="val"
        )
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
        _, _, holdout_preds = valid_fn(holdout_loader, model, device)
        holdout_preds_folds.append(holdout_preds)

        # Inference on Test Set
        test_dataset = PhraseDataset(test_df, tokenizer, mode="test", cache_name="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
        _, _, test_preds = valid_fn(test_loader, model, device)
        test_preds_folds.append(test_preds)

        # Cleanup to free GPU memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            holdout_loader,
            test_loader,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Ensemble & Evaluation
    print(f"\n{'='*20} Ensemble Evaluation {'='*20}")

    # Average predictions across folds
    avg_holdout_preds = np.mean(holdout_preds_folds, axis=0)
    avg_test_preds = np.mean(test_preds_folds, axis=0)

    # Metric on Holdout
    final_metric = get_score(holdout_val_df["score"].values, avg_holdout_preds)
    # Required output format
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")
    # Calculate absolute errors
    errors = np.abs(holdout_val_df["score"].values - avg_holdout_preds)

    # Get structural features for holdout set (reuse function)
    # feats columns: [levenshtein, jaccard, len_ratio]
    feats = get_features_batch(
        holdout_val_df["anchor"], holdout_val_df["target"], cache_name="val"
    )

    lev_corr = get_score(errors, feats[:, 0])
    jac_corr = get_score(errors, feats[:, 1])
    len_corr = get_score(errors, feats[:, 2])

    print(f"Correlation of Error with Normalized Levenshtein: {lev_corr:.4f}")
    print(f"Correlation of Error with Jaccard Similarity: {jac_corr:.4f}")
    print(f"Correlation of Error with Length Ratio: {len_corr:.4f}")

    # 7. Submission
    threshold = 0.8654320295612139
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        submission = pd.DataFrame({"id": test_df["id"], "score": avg_test_preds})
        submission.to_csv(CFG.submission_file, index=False)
        print(f"Submission saved to {CFG.submission_file}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
