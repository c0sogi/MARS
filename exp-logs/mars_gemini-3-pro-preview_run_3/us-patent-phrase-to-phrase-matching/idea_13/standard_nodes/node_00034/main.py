import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.dataset import process_data, get_dataloaders, get_test_dataloader
from library.model import CustomModel
from library.loss import HybridLoss
from library.training_utils import AWP, EMA, get_optimizer_params
from library.engine import train_fn, valid_fn, inference_fn


def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_failure_analysis(val_df, preds, targets):
    """
    Analyzes the correlation between error magnitude and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    val_df = val_df.copy()
    val_df["pred"] = preds
    val_df["target"] = targets
    val_df["abs_error"] = (val_df["pred"] - val_df["target"]).abs()

    # Feature extraction
    val_df["anchor_len"] = val_df["anchor"].astype(str).apply(len)
    val_df["target_len"] = val_df["target"].astype(str).apply(len)
    val_df["context_len"] = val_df["context_text"].astype(str).apply(len)

    # Correlation analysis
    features = ["anchor_len", "target_len", "context_len"]
    correlations = {}
    for feat in features:
        corr = val_df["abs_error"].corr(val_df[feat])
        correlations[feat] = corr

    print("Correlation between Absolute Error and Input Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    return correlations


def main():
    # 1. Setup and Configuration
    # Override defaults for a fast baseline execution (1 hour limit)
    # We run 1 fold for 2 epochs.
    cfg = Config(
        epochs=2,
        trn_folds=[0],
        train_batch_size=8,
        valid_batch_size=16,
        print_freq=100,
        num_workers=2,
    )

    seed_everything(cfg.seed)

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    print(f"Configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Folds to train: {cfg.trn_folds}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Device: {cfg.device}")

    # 2. Data Preparation
    print("\nProcessing Data...")
    # This will load metadata, merge context, and create folds
    full_df = process_data(cfg, load_cached_data=True)

    # 3. Training Loop
    best_pearson_global = -1.0

    # Store OOF data for global analysis (though we only run 1 fold here)
    oof_ids = []
    oof_preds = []
    oof_targets = []

    for fold in cfg.trn_folds:
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # DataLoaders
        train_loader, val_loader = get_dataloaders(cfg, fold, load_cached_data=True)

        # Model
        model = CustomModel(cfg)
        model.to(cfg.device)

        # Optimizer
        optimizer_params = get_optimizer_params(model, cfg)
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas
        )

        # Scheduler
        num_training_steps = len(train_loader) * cfg.epochs
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=cfg.num_cycles,
        )

        # Loss
        criterion = HybridLoss(cfg)

        # AWP & EMA
        awp = (
            AWP(
                model,
                optimizer,
                adv_lr=cfg.awp_lr,
                adv_eps=cfg.awp_eps,
                start_epoch=cfg.awp_start_epoch,
            )
            if cfg.use_awp
            else None
        )

        ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        # Training
        best_pearson = -1.0
        best_model_path = os.path.join(cfg.working_dir, f"model_fold{fold}.pth")

        for epoch in range(cfg.epochs):
            # Train
            avg_loss = train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                cfg.device,
                cfg,
                awp,
                ema,
            )

            # Validate
            val_loss, val_pearson, val_preds = valid_fn(
                fold, val_loader, model, criterion, cfg.device, cfg, ema
            )

            print(
                f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Pearson: {val_pearson:.4f}"
            )

            # Save Best
            if val_pearson > best_pearson:
                print(
                    f"Score Improved ({best_pearson:.4f} -> {val_pearson:.4f}). Saving model..."
                )
                best_pearson = val_pearson
                # Save state dict (use EMA weights if available)
                if ema is not None:
                    ema.apply_shadow()
                torch.save(model.state_dict(), best_model_path)
                if ema is not None:
                    ema.restore()

        print(f"Fold {fold} Best Pearson: {best_pearson:.4f}")

        # Load best model for OOF generation / Inference
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))

        # Generate OOF for this fold
        _, final_pearson, final_preds = valid_fn(
            fold, val_loader, model, criterion, cfg.device, cfg, ema=None
        )

        # Get targets for this fold
        val_df_fold = full_df[full_df["fold"] == fold].reset_index(drop=True)
        # Debug subsetting in get_dataloaders might truncate val_df,
        # but here we assume full run or consistent debug size.
        # To be safe, we extract targets from loader or match length.
        # Ideally, valid_fn returns predictions aligned with the loader.
        # We'll use the targets collected inside valid_fn if we had returned them,
        # but we can also just take the first N rows of val_df_fold matching preds.
        val_df_fold = val_df_fold.iloc[: len(final_preds)]

        oof_ids.extend(val_df_fold["id"].values)
        oof_preds.extend(final_preds)
        oof_targets.extend(val_df_fold["score"].values)

        best_pearson_global = max(best_pearson_global, best_pearson)

    # 4. Global Validation & Failure Analysis
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Compute Final Metric
    final_metric = np.corrcoef(oof_preds, oof_targets)[0, 1]
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Reconstruct a dataframe for the analyzed samples
    analyzed_df = full_df[full_df["id"].isin(oof_ids)].copy().reset_index(drop=True)
    # Ensure alignment is correct by mapping IDs
    # Create a map from ID to pred
    pred_map = dict(zip(oof_ids, oof_preds))
    target_map = dict(zip(oof_ids, oof_targets))

    analyzed_df["pred"] = analyzed_df["id"].map(pred_map)
    analyzed_df["target_check"] = analyzed_df["id"].map(target_map)

    run_failure_analysis(analyzed_df, analyzed_df["pred"], analyzed_df["target_check"])

    # 5. Submission
    threshold = 0.8698034882545471
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        test_loader = get_test_dataloader(cfg, load_cached_data=True)

        # We need to ensemble predictions from all trained folds.
        # In this fast baseline, we likely only have Fold 0.
        fold_preds = []

        for fold in cfg.trn_folds:
            model_path = os.path.join(cfg.working_dir, f"model_fold{fold}.pth")
            model = CustomModel(cfg)
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            model.to(cfg.device)

            ids, preds = inference_fn(test_loader, model, cfg.device, cfg, ema=None)
            fold_preds.append(preds)

        # Average predictions
        avg_preds = np.mean(fold_preds, axis=0)

        # Create submission dataframe
        submission_df = pd.DataFrame({"id": ids, "score": avg_preds})

        # Save
        sub_path = "./submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
