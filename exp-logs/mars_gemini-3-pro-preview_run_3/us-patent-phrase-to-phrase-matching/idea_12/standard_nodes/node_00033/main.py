import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from library
from library.config import CFG
from library.utils import seed_everything, get_logger, AWP, EMA, get_score
from library.dataset import prepare_data, CPCDataset
from library.model import DebertaV3Model
from library.engine import train_fn, valid_fn, get_optimizer_params
from library.loss import HybridLoss


def run_failure_analysis(df_val, preds, logger):
    """
    Analyzes the correlation between prediction errors and input features.
    """
    logger.info("Starting Failure Analysis...")
    df_val = df_val.copy()
    df_val["pred"] = preds
    df_val["abs_error"] = (df_val["score"] - df_val["pred"]).abs()

    # Feature Engineering for Analysis
    df_val["anchor_len"] = df_val["anchor"].astype(str).apply(len)
    df_val["target_len"] = df_val["target"].astype(str).apply(len)

    # Context Frequency
    context_counts = df_val["context"].value_counts().to_dict()
    df_val["context_freq"] = df_val["context"].map(context_counts)

    # Correlations
    features = ["score", "anchor_len", "target_len", "context_freq"]
    correlations = {}
    for feat in features:
        if feat in df_val.columns:
            # Handle potential NaNs or constant values
            if df_val[feat].nunique() > 1:
                corr, _ = pearsonr(df_val[feat], df_val["abs_error"])
                correlations[feat] = corr
            else:
                correlations[feat] = 0.0

    logger.info("Correlation between Error Magnitude and Features:")
    for feat, corr in correlations.items():
        logger.info(f"  {feat}: {corr:.4f}")


def inference(model, loader, device):
    """
    Runs inference on a dataloader (no labels expected).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs in loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # Model returns dict
            outputs = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            score = outputs["score"].view(-1).cpu().numpy()
            preds.append(score)

    return np.concatenate(preds)


def main():
    # 1. Setup & Configuration Override
    # We override specific configs to ensure the run finishes within the time limit
    # while still training a robust ensemble.
    CFG.epochs = 2
    CFG.trn_fold = [0, 1]  # Train first 2 folds for the baseline ensemble
    CFG.debug = False  # Use full data for the selected folds

    seed_everything(CFG.seed)

    # Create output directories
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(CFG.submission_path), exist_ok=True)

    logger = get_logger(os.path.join(CFG.output_dir, "run.log"))
    logger.info(
        f"Configuration: Epochs={CFG.epochs}, Folds={CFG.trn_fold}, Device={CFG.device}"
    )

    # 2. Data Loading
    logger.info("Loading Data...")
    # Load metadata files (already split)
    train_df = prepare_data(CFG, split="train")
    val_df = prepare_data(CFG, split="val")
    test_df = prepare_data(CFG, split="test")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # 3. Stratified K-Fold Setup
    # We apply StratifiedKFold on the training set to create internal validation splits
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    train_df["fold"] = -1
    for i, (_, val_idx) in enumerate(
        skf.split(train_df, train_df["score"].astype(str))
    ):
        train_df.loc[val_idx, "fold"] = i

    # 4. Training Loop
    best_models = []

    for fold in CFG.trn_fold:
        logger.info(f"=== Starting Fold {fold} ===")

        # Split Train/Internal Val
        df_trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
        df_val_internal = train_df[train_df["fold"] == fold].reset_index(drop=True)

        # Datasets
        train_dataset = CPCDataset(df_trn, tokenizer, CFG, mode="train")
        valid_dataset = CPCDataset(df_val_internal, tokenizer, CFG, mode="train")

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Model Initialization
        model = DebertaV3Model(CFG, pretrained=True)
        model.to(CFG.device)

        # Optimizer & Scheduler (LLRD)
        optimizer_params = get_optimizer_params(
            model,
            encoder_lr=CFG.encoder_lr,
            decoder_lr=CFG.head_lr,
            weight_decay=CFG.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
        )

        num_training_steps = len(train_loader) * CFG.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * CFG.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        # Loss, AWP, EMA
        criterion = HybridLoss(CFG)

        awp = None
        if CFG.awp:
            awp = AWP(
                model,
                optimizer,
                adv_lr=CFG.awp_lr,
                adv_eps=CFG.awp_eps,
                start_epoch=CFG.awp_start_epoch,
            )

        ema = None
        if CFG.ema:
            ema = EMA(model, decay=CFG.ema_decay)
            ema.register()

        # Training
        best_score = -1
        best_model_path = os.path.join(CFG.output_dir, f"model_fold{fold}.pth")

        for epoch in range(CFG.epochs):
            avg_loss = train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                CFG.device,
                awp,
                ema,
                CFG,
            )

            # Validation (using EMA weights if enabled)
            if ema:
                ema.apply_shadow()

            val_loss, val_score = valid_fn(
                valid_loader, model, criterion, CFG.device, CFG
            )

            logger.info(
                f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}"
            )

            # Save Best Model
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New Best Score for Fold {fold}! Model saved.")

            if ema:
                ema.restore()

        best_models.append(best_model_path)

        # Cleanup to free memory
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

    # 5. Final Validation on Hold-out Set (val.csv)
    logger.info("=== Final Validation on Hold-out Set ===")

    # Use mode='test' to get inputs only; we have labels in df
    val_dataset_final = CPCDataset(val_df, tokenizer, CFG, mode="test")
    val_loader_final = DataLoader(
        val_dataset_final,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    fold_preds = []
    for model_path in best_models:
        model = DebertaV3Model(CFG, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        model.to(CFG.device)

        preds = inference(model, val_loader_final, CFG.device)
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Ensemble: Average Predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Clip predictions to valid range
    avg_preds = np.clip(avg_preds, 0, 1)

    # Compute Final Metric
    final_score = get_score(val_df["score"].values, avg_preds)
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    run_failure_analysis(val_df, avg_preds, logger)

    # 7. Submission Generation
    threshold = 0.8698034882545471
    if final_score > threshold:
        logger.info(f"Score {final_score} > {threshold}. Generating submission...")

        test_dataset = CPCDataset(test_df, tokenizer, CFG, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size * 2,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        test_fold_preds = []
        for model_path in best_models:
            model = DebertaV3Model(CFG, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=CFG.device))
            model.to(CFG.device)

            preds = inference(model, test_loader, CFG.device)
            test_fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        test_avg_preds = np.mean(test_fold_preds, axis=0)
        test_avg_preds = np.clip(test_avg_preds, 0, 1)

        submission = pd.DataFrame({"id": test_df["id"], "score": test_avg_preds})

        submission.to_csv(CFG.submission_path, index=False)
        logger.info(f"Submission saved to {CFG.submission_path}")

    else:
        logger.info(
            f"Score {final_score} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
