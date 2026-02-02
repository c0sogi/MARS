import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_logger, AverageMeter
from library.cpc_mapping import get_cpc_texts
from library.dataset import PearsonDataset
from library.model import CustomModel
from library.engine import train_fn, valid_fn, get_optimizer_params


def main():
    # 1. Configuration & Setup
    cfg = Config()

    # Adjustments for Fast Baseline on A100
    cfg.folds = 3  # Reduce folds to 3 for speed
    cfg.epochs = 2  # Reduce epochs to 2
    cfg.train_batch_size = 16  # Increase batch size for A100
    cfg.valid_batch_size = 32

    # Setup Output Directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.predictions_dir, exist_ok=True)

    logger = get_logger(os.path.join(cfg.output_dir, "train.log"))
    seed_everything(cfg.seed)

    logger.info("Starting Fast Baseline Workflow...")
    logger.info(f"Device: {cfg.device}")

    # 2. Data Loading
    logger.info("Loading Metadata...")
    df_train_full = pd.read_csv(cfg.train_path)
    df_val_holdout = pd.read_csv(cfg.val_path)
    df_test = pd.read_csv(cfg.test_path)

    # Load CPC Descriptions
    cpc_texts = get_cpc_texts(cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 3. Cross-Validation Training
    # We split df_train_full into K folds
    sgkf = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    # Create fold column
    df_train_full["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df_train_full, df_train_full["score"], df_train_full["anchor"])
    ):
        df_train_full.loc[val_idx, "fold"] = fold_idx

    # Store paths to saved models for ensemble
    model_paths = []

    for fold in range(cfg.folds):
        logger.info(f"\n{'='*20} Fold {fold+1}/{cfg.folds} {'='*20}")

        # Split Data
        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        valid_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        # Subsample for extremely fast baseline if needed (optional, but keeping full data for quality)
        # To ensure < 2 hours, we rely on A100 speed + reduced epochs/folds.

        # Datasets
        train_dataset = PearsonDataset(
            train_df, tokenizer, cpc_texts, f"train_fold_{fold}", cfg
        )
        valid_dataset = PearsonDataset(
            valid_df, tokenizer, cpc_texts, f"valid_fold_{fold}", cfg
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # Model
        model = CustomModel(cfg, pretrained=True)
        model.to(cfg.device)

        # Optimizer with LLRD
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=cfg.learning_rate,
            decoder_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_parameters, eps=1e-6)

        # Scheduler
        num_train_steps = int(len(train_df) / cfg.train_batch_size * cfg.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Training Loop
        best_pearson = -1.0
        best_model_path = os.path.join(cfg.model_dir, f"model_fold_{fold}.pth")

        for epoch in range(cfg.epochs):
            train_loss = train_fn(
                train_loader,
                model,
                optimizer,
                epoch,
                scheduler,
                cfg.device,
                cfg,
                logger,
            )
            val_loss, val_pearson, _ = valid_fn(
                valid_loader, model, cfg.device, cfg, logger
            )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"Saved Best Model for Fold {fold} with Pearson: {best_pearson:.4f}"
                )

        model_paths.append(best_model_path)

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        torch.cuda.empty_cache()

    # 4. Hold-out Validation (Ensemble)
    logger.info(f"\n{'='*20} Hold-out Validation {'='*20}")

    val_dataset = PearsonDataset(
        df_val_holdout, tokenizer, cpc_texts, "holdout_val", cfg
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    ensemble_preds = []

    for path in model_paths:
        logger.info(f"Inference with model: {os.path.basename(path)}")
        model = CustomModel(cfg, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        model.to(cfg.device)

        _, _, preds = valid_fn(val_loader, model, cfg.device, cfg, logger=None)
        ensemble_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    avg_preds = np.mean(ensemble_preds, axis=0)

    # Calculate Metric
    final_targets = df_val_holdout["score"].values
    final_pearson, _ = pearsonr(final_targets, avg_preds)

    print(f"Final Validation Metric: {final_pearson}")

    # 5. Failure Analysis
    logger.info(f"\n{'='*20} Failure Analysis {'='*20}")

    df_analysis = df_val_holdout.copy()
    df_analysis["pred"] = avg_preds
    df_analysis["error"] = np.abs(df_analysis["score"] - df_analysis["pred"])

    # Feature Engineering for Analysis
    df_analysis["len_anchor"] = df_analysis["anchor"].astype(str).apply(len)
    df_analysis["len_target"] = df_analysis["target"].astype(str).apply(len)
    df_analysis["len_diff"] = np.abs(
        df_analysis["len_anchor"] - df_analysis["len_target"]
    )

    # Correlations
    correlations = (
        df_analysis[["error", "score", "len_anchor", "len_target", "len_diff"]]
        .corr()["error"]
        .sort_values(ascending=False)
    )
    logger.info("Correlation of Error with Features:")
    logger.info(correlations)

    # 6. Submission
    THRESHOLD = 0.8673
    if final_pearson > THRESHOLD:
        logger.info(
            f"\nValidation Metric {final_pearson:.4f} > {THRESHOLD}. Generating Submission..."
        )

        test_dataset = PearsonDataset(df_test, tokenizer, cpc_texts, "test", cfg)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        test_ensemble_preds = []

        for path in model_paths:
            model = CustomModel(cfg, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=cfg.device))
            model.to(cfg.device)

            _, _, preds = valid_fn(test_loader, model, cfg.device, cfg, logger=None)
            test_ensemble_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_ensemble_preds, axis=0)

        submission = pd.DataFrame({"id": df_test["id"], "score": avg_test_preds})

        submission.to_csv(cfg.submission_path, index=False)
        logger.info(f"Submission saved to {cfg.submission_path}")

    else:
        logger.info(
            f"\nValidation Metric {final_pearson:.4f} <= {THRESHOLD}. Skipping Submission."
        )


if __name__ == "__main__":
    main()
