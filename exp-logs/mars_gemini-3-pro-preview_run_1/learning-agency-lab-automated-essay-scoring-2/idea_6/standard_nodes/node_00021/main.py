import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, quadratic_weighted_kappa
from library.dataset import get_mlm_loader, EssayDataset, SmartCollator, load_dataframes
from library.model import EssayModel
from library.engine import (
    train_mlm,
    train_fn,
    valid_fn,
    inference_fn,
    get_optimizer_params,
)
from library.awp import AWP


def run():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger()

    # Override Config for Fast Baseline to ensure < 2 hours runtime
    # DeBERTa-Large on A100 is fast, but 5 folds * 4 epochs is too much.
    # We reduce to 1 epoch per stage for the baseline verification.
    Config.mlm_epochs = 1
    Config.epochs = 1
    Config.awp_start_epoch = 0  # Enable AWP immediately since we only run 1 epoch
    Config.debug = False

    # Ensure batch sizes are optimized for A100 (40GB)
    Config.train_batch_size = 4
    Config.gradient_accumulation_steps = 4
    Config.valid_batch_size = 8

    logger.info("Configuration set for fast baseline execution.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 2. MLM Stage: Domain-Adaptive Pre-training
    logger.info("--- Starting Stage 1: Domain-Adaptive Pre-training (MLM) ---")
    # Load combined corpus (Train + Val + Test)
    mlm_loader = get_mlm_loader(tokenizer, load_cached_data=True)
    # Train MLM and save to Config.mlm_model_dir
    train_mlm(mlm_loader)

    # 3. SFT Stage: Supervised Fine-Tuning with 5-Fold CV
    logger.info("--- Starting Stage 2: Supervised Fine-Tuning (5-Fold CV) ---")

    # Load training data explicitly to handle CV splits
    df_train = pd.read_csv(Config.train_path)
    df_train[Config.text_col] = df_train[Config.text_col].fillna("").astype(str)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    fold_models = []
    os.makedirs(Config.working_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train[Config.target_col])
    ):
        logger.info(f"--- Fold {fold + 1}/{Config.n_folds} ---")

        # Create Fold Dataframes
        fold_train_df = df_train.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_train.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = EssayDataset(
            fold_train_df, tokenizer, Config.max_length, is_test=False
        )
        val_dataset = EssayDataset(
            fold_val_df, tokenizer, Config.max_length, is_test=False
        )

        # Create DataLoaders
        collator = SmartCollator(tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=False,
        )

        # Initialize Model (Load from MLM checkpoint)
        # pretrained=True loads the backbone weights from the MLM checkpoint
        model = EssayModel(checkpoint_path=Config.mlm_model_dir, pretrained=True)
        model.to(Config.device)

        # Optimizer with Layer-wise Learning Rate Decay
        optimizer_params = get_optimizer_params(
            model,
            learning_rate=Config.learning_rate,
            weight_decay=Config.weight_decay,
            llrd_decay=Config.llrd_decay,
        )
        optimizer = AdamW(optimizer_params)

        # Scheduler
        num_training_steps = len(train_loader) * Config.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        # Adversarial Weight Perturbation (AWP)
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

        # Training Loop
        best_score = -1.0
        best_model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                model, train_loader, optimizer, scheduler, epoch, awp=awp
            )
            val_loss, val_score = valid_fn(model, val_loader)

            logger.info(
                f"Fold {fold+1} Epoch {epoch+1} - Train Loss: {avg_loss:.4f} Val Loss: {val_loss:.4f} Val QWK: {val_score:.4f}"
            )

            # Save best model (always save for 1-epoch baseline)
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)

        # Cleanup to free memory
        del (
            model,
            optimizer,
            scheduler,
            awp,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()
        fold_models.append(best_model_path)

    # 4. Final Validation on Hold-out Set
    logger.info("--- Final Validation on Hold-out Set ---")
    df_val_holdout = pd.read_csv(Config.val_path)
    df_val_holdout[Config.text_col] = (
        df_val_holdout[Config.text_col].fillna("").astype(str)
    )

    val_dataset = EssayDataset(
        df_val_holdout, tokenizer, Config.max_length, is_test=False
    )
    collator = SmartCollator(tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Ensemble Inference
    all_preds = []
    for model_path in fold_models:
        # Initialize model structure without loading pre-trained weights (faster)
        model = EssayModel(checkpoint_path=None, pretrained=False)
        # Load fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)

        preds = inference_fn(model, val_loader)
        all_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_preds = np.mean(all_preds, axis=0)

    # Compute Metric
    true_scores = df_val_holdout[Config.target_col].values
    final_score = quadratic_weighted_kappa(true_scores, avg_preds)

    print(f"Final Validation Metric: {final_score}")

    # 5. Failure Analysis
    logger.info("--- Failure Analysis ---")
    # Round predictions to integers for error analysis
    rounded_preds = np.round(np.clip(avg_preds, 1, 6)).astype(int)
    errors = np.abs(true_scores - rounded_preds)

    # Calculate text length
    lengths = df_val_holdout[Config.text_col].apply(len).values

    # Compute correlation
    corr, _ = pearsonr(errors, lengths)
    print(f"Correlation between Error and Text Length: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.8274925140324321
    if final_score > THRESHOLD:
        logger.info("--- Generating Submission ---")
        df_test = pd.read_csv(Config.test_path)
        df_test[Config.text_col] = df_test[Config.text_col].fillna("").astype(str)

        test_dataset = EssayDataset(df_test, tokenizer, Config.max_length, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        test_preds_list = []
        for model_path in fold_models:
            model = EssayModel(checkpoint_path=None, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            model.to(Config.device)

            p = inference_fn(model, test_loader)
            test_preds_list.append(p)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_preds_list, axis=0)
        final_test_preds = np.round(np.clip(avg_test_preds, 1, 6)).astype(int)

        submission = pd.DataFrame(
            {Config.id_col: df_test[Config.id_col], Config.target_col: final_test_preds}
        )

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(
            f"Score {final_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
