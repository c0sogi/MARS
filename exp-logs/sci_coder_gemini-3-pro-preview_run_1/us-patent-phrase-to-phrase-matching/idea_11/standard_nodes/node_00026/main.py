import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metrics
from library.data import (
    get_data,
    get_dataloaders,
    get_test_dataloader,
    PatentDataset,
    Collate,
)
from library.model import PatentModel, get_optimizer_params
from library.engine import train_fn, valid_fn, inference_fn

# Suppress HF warnings for cleaner output
hf_logging.set_verbosity_error()


def main():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger()
    device = Config.device

    # Adjust Config for Fast Baseline within 2 hours
    # Reducing to 4 folds * 3 epochs ensures completion within ~75-90 mins on A100
    Config.epochs = 3
    Config.num_folds = 4

    logger.info(
        f"Starting run with {Config.num_folds} folds and {Config.epochs} epochs."
    )

    # 2. Data Loading
    # train_df: metadata/train.csv (Used for Cross-Validation)
    # val_df: metadata/val.csv (Strict Hold-out for Final Metric)
    # test_df: metadata/test.csv (For Submission)
    train_df, val_df, test_df = get_data(load_cached_data=True)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 4. Prepare Holdout & Test Loaders
    # We will predict on these every fold and average the results
    val_loader_holdout = get_test_dataloader(val_df, tokenizer)
    test_loader = get_test_dataloader(test_df, tokenizer)

    # Arrays to store ensemble predictions
    val_preds_ensemble = np.zeros((len(val_df), Config.num_folds))
    test_preds_ensemble = np.zeros((len(test_df), Config.num_folds))

    # 5. Cross-Validation Loop
    sgkf = StratifiedGroupKFold(n_splits=Config.num_folds)

    # Split train_df using StratifiedGroupKFold
    fold = 0
    for train_idx, valid_idx in sgkf.split(
        train_df, train_df[Config.target_col], train_df[Config.group_col]
    ):
        logger.info(f"--- Starting Fold {fold+1}/{Config.num_folds} ---")

        # Create Fold Splits
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

        # Create DataLoaders for this fold
        train_loader, valid_loader = get_dataloaders(
            fold_train_df, fold_valid_df, tokenizer
        )

        # Initialize Model
        model = PatentModel(Config.model_name)
        model.to(device)

        # Optimizer & Scheduler with LLRD
        optimizer_params = get_optimizer_params(
            model,
            base_lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
            llrd_decay=Config.llrd_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_params, lr=Config.learning_rate)

        num_training_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training Loop
        for epoch in range(Config.epochs):
            # Train
            train_loss = train_fn(
                model, train_loader, optimizer, scheduler, device, epoch
            )

            # Validation (Internal Fold Monitor)
            val_loss, metrics = valid_fn(model, valid_loader, device)
            logger.info(
                f"Fold {fold+1} Epoch {epoch+1}: Loss {val_loss:.4f}, Pearson {metrics['pearson']:.4f}"
            )

        # Inference on Holdout Val and Test with the trained model
        logger.info(f"Fold {fold+1} Inference...")
        val_preds_fold = inference_fn(model, val_loader_holdout, device)
        test_preds_fold = inference_fn(model, test_loader, device)

        val_preds_ensemble[:, fold] = val_preds_fold
        test_preds_ensemble[:, fold] = test_preds_fold

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

        fold += 1

    # 6. Aggregation & Evaluation
    logger.info("Aggregating predictions...")

    # Average predictions across folds
    final_val_preds = np.mean(val_preds_ensemble, axis=1)
    final_test_preds = np.mean(test_preds_ensemble, axis=1)

    # Compute Final Metric on Holdout
    labels = val_df[Config.target_col].values
    metrics = compute_metrics((final_val_preds, labels))
    final_metric = metrics["pearson"]

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_df["predicted_score"] = final_val_preds
    val_df["error"] = (val_df["score"] - val_df["predicted_score"]).abs()

    # Create features for correlation analysis
    val_df["anchor_len"] = val_df["anchor"].astype(str).apply(len)
    val_df["target_len"] = val_df["target"].astype(str).apply(len)

    # Calculate correlations
    analysis_cols = ["error", "score", "anchor_len", "target_len"]
    corrs = val_df[analysis_cols].corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(corrs)

    # 8. Submission
    threshold = 0.8673
    if final_metric > threshold:
        logger.info(f"Metric {final_metric:.4f} > {threshold}. Generating submission.")

        submission_df = pd.DataFrame({"id": test_df["id"], "score": final_test_preds})

        # Ensure output directory exists
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        logger.info(f"Metric {final_metric:.4f} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
