import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.cpc_mapping import get_cpc_texts
from library.data import (
    get_data_splits,
    prepare_loaders,
    prepare_test_loader,
    PhraseDataset,
    preprocess_test_data,
)
from library.model import CustomModel
from library.training_utils import (
    get_optimizer_params,
    train_fn,
    valid_fn,
    inference_fn,
)


def run():
    # 1. Setup
    cfg = Config()

    # Overrides for Fast Baseline execution to fit within time limits
    cfg.epochs = 2
    cfg.n_folds = 4
    cfg.train_batch_size = 16
    cfg.valid_batch_size = 32
    # Ensure gradient checkpointing is on to fit batch size 16 on A100 with Large model
    cfg.gradient_checkpointing = True

    # Setup directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.predictions_dir, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)

    # Logging
    logger = get_logger(os.path.join(cfg.output_dir, "main_script.log"))
    seed_everything(cfg.seed)

    logger.info("Configuration:")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Epochs: {cfg.epochs}, Folds: {cfg.n_folds}")
    logger.info(f"Device: {cfg.device}")

    # 2. Data Preparation
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Get Train Data (with folds)
    # get_data_splits handles caching and CPC mapping internally
    train_df = get_data_splits(cfg, load_cached_data=True)

    # 3. Training Loop
    model_paths = []

    for fold in range(cfg.n_folds):
        logger.info(f"=== Starting Fold {fold} ===")

        # Prepare Loaders
        train_loader, valid_loader = prepare_loaders(fold, train_df, tokenizer, cfg)

        # Initialize Model
        model = CustomModel(cfg, pretrained=True)
        model.to(cfg.device)

        # Optimizer & Scheduler
        optimizer_parameters = get_optimizer_params(model, cfg)
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=cfg.learning_rate, eps=cfg.eps, betas=cfg.betas
        )

        num_train_steps = int(len(train_loader) * cfg.epochs)
        num_warmup_steps = int(num_train_steps * cfg.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles,
        )

        # Training
        best_pearson = -1.0
        best_model_path = os.path.join(cfg.model_dir, f"model_fold_{fold}.pth")

        for epoch in range(cfg.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch, scheduler, cfg.device, cfg
            )
            val_loss, val_pearson = valid_fn(valid_loader, model, cfg.device, cfg)

            logger.info(
                f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_pearson:.4f}"
            )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"New best model saved for fold {fold} with Pearson: {best_pearson:.4f}"
                )

        model_paths.append(best_model_path)

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Validation on Hold-out Set
    logger.info("=== Performing Final Validation on Hold-out Set ===")

    # Load Hold-out Validation Data
    val_df = pd.read_csv(cfg.val_path)

    # Map CPC codes for validation set
    cpc_texts = get_cpc_texts(cfg, load_cached_data=True)
    val_df["context_text"] = val_df["context"].map(cpc_texts).fillna("")

    # Create Dataset/Loader
    val_dataset = PhraseDataset(val_df, tokenizer, cfg.max_length, is_test=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Ensemble Inference
    fold_preds = []
    for fold, path in enumerate(model_paths):
        logger.info(f"Predicting with model fold {fold}...")
        model = CustomModel(cfg, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        model.to(cfg.device)

        preds = inference_fn(val_loader, model, cfg.device)
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Calculate Metric
    labels = val_df["score"].values
    final_metric, _ = pearsonr(labels, avg_preds)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("=== Failure Analysis ===")
    val_df["pred"] = avg_preds
    val_df["error"] = np.abs(val_df["score"] - val_df["pred"])

    # Feature Engineering for Analysis
    val_df["len_anchor"] = val_df["anchor"].astype(str).apply(len)
    val_df["len_target"] = val_df["target"].astype(str).apply(len)
    val_df["len_diff"] = np.abs(val_df["len_anchor"] - val_df["len_target"])

    # Correlations
    analysis_cols = ["len_anchor", "len_target", "len_diff", "score"]
    print("Correlation between Error Magnitude and Features:")
    for col in analysis_cols:
        corr, _ = pearsonr(val_df["error"], val_df[col])
        print(f"{col}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.8673
    if final_metric > THRESHOLD:
        logger.info(
            f"Metric {final_metric:.4f} > {THRESHOLD}. Generating submission..."
        )

        # Load and Process Test Data
        test_df = preprocess_test_data(cfg, load_cached_data=True)
        test_loader = prepare_test_loader(test_df, tokenizer, cfg)

        test_fold_preds = []
        for fold, path in enumerate(model_paths):
            logger.info(f"Inference on Test Set with Fold {fold}...")
            model = CustomModel(cfg, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=cfg.device))
            model.to(cfg.device)

            preds = inference_fn(test_loader, model, cfg.device)
            test_fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()
            gc.collect()

        avg_test_preds = np.mean(test_fold_preds, axis=0)

        # Create Submission File
        submission = pd.DataFrame({"id": test_df["id"], "score": avg_test_preds})

        submission.to_csv(cfg.submission_path, index=False)
        logger.info(f"Submission saved to {cfg.submission_path}")

    else:
        logger.info(f"Metric {final_metric:.4f} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
