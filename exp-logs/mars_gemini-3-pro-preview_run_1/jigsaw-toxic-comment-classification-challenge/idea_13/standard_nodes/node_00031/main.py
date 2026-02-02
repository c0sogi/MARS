import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, get_score
from library.dataset import get_data, MLMDataset, ToxicDataset
from library.model import DeepSupervisedModel, AWP
from library.engine import train_mlm, train_fn, valid_fn, inference_fn


def run():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    # Override Config for Fast Baseline (2-hour limit)
    Config.epochs = 2
    Config.train_batch_size = 32
    Config.valid_batch_size = 64
    Config.awp_start_epoch = 1  # Start AWP earlier since we have fewer epochs

    # Subsampling limits
    TRAIN_SAMPLE_SIZE = 50000
    TEST_SUBSET_SIZE = 20000  # For distillation soft labels

    seed_everything(Config.seed)
    logger = get_logger(os.path.join(Config.working_dir, "run.log"))
    logger.info("Starting End-to-End Pipeline...")

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    logger.info("Loading Data...")
    train_df, val_df, test_df = get_data(load_cached_data=True)

    # Subsample Train and Test for speed
    if len(train_df) > TRAIN_SAMPLE_SIZE:
        logger.info(f"Subsampling Train from {len(train_df)} to {TRAIN_SAMPLE_SIZE}")
        train_df = train_df.sample(
            n=TRAIN_SAMPLE_SIZE, random_state=Config.seed
        ).reset_index(drop=True)

    # Create a subset of test for distillation (generating soft labels)
    if len(test_df) > TEST_SUBSET_SIZE:
        test_subset = test_df.sample(
            n=TEST_SUBSET_SIZE, random_state=Config.seed
        ).reset_index(drop=True)
    else:
        test_subset = test_df.copy()

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # ==========================================
    # 3. Stage 1: Domain-Adaptive Pre-training (DAPT)
    # ==========================================
    logger.info("\n=== Stage 1: Domain-Adaptive Pre-training (DAPT) ===")

    # Combine text for MLM
    dapt_text = pd.concat(
        [train_df["comment_text"], val_df["comment_text"], test_subset["comment_text"]],
        axis=0,
    ).reset_index(drop=True)

    # Create Dataset & Loader
    dapt_dataset = MLMDataset(
        pd.DataFrame({"comment_text": dapt_text}), tokenizer, Config.max_len
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dapt_loader = DataLoader(
        dapt_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # Initialize MLM Model
    dapt_model = AutoModelForMaskedLM.from_pretrained(Config.model_name).to(
        Config.device
    )
    optimizer = AdamW(
        dapt_model.parameters(), lr=5e-5, weight_decay=Config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=5e-5, total_steps=len(dapt_loader), pct_start=0.1
    )

    # Train DAPT (1 Epoch is sufficient for adaptation)
    train_mlm(dapt_model, dapt_loader, optimizer, scheduler, Config.device, 0, logger)

    # Save Backbone
    dapt_backbone_path = os.path.join(Config.working_dir, "dapt_backbone.pth")
    # Save the base model weights (e.g., 'deberta' from 'DebertaV3ForMaskedLM')
    if hasattr(dapt_model, "base_model"):
        torch.save(dapt_model.base_model.state_dict(), dapt_backbone_path)
    else:
        # Fallback for models where base_model attribute might not be direct
        torch.save(dapt_model.state_dict(), dapt_backbone_path)

    logger.info(f"DAPT Backbone saved to {dapt_backbone_path}")

    # Cleanup
    del dapt_model, optimizer, scheduler, dapt_loader, dapt_dataset, dapt_text
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # 4. Stage 2: Teacher Training
    # ==========================================
    logger.info("\n=== Stage 2: Teacher Training ===")

    # Datasets
    train_ds = ToxicDataset(train_df, tokenizer, Config.max_len)
    val_ds = ToxicDataset(val_df, tokenizer, Config.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model Setup
    teacher_model = DeepSupervisedModel(pretrained=True).to(Config.device)
    # Load DAPT weights
    try:
        msg = teacher_model.model.load_state_dict(
            torch.load(dapt_backbone_path), strict=False
        )
        logger.info(f"Teacher loaded DAPT weights: {msg}")
    except Exception as e:
        logger.warning(f"Failed to load DAPT weights strictly: {e}")

    optimizer = AdamW(
        teacher_model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        total_steps=len(train_loader) * Config.epochs,
        pct_start=Config.pct_start,
    )
    criterion = nn.BCEWithLogitsLoss()
    awp = (
        AWP(teacher_model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)
        if Config.use_awp
        else None
    )

    # Train Teacher
    for epoch in range(Config.epochs):
        train_fn(
            0,
            train_loader,
            teacher_model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            Config.device,
            awp,
            logger,
        )
        valid_fn(val_loader, teacher_model, criterion, Config.device, logger)

    # Generate Soft Labels on Test Subset
    logger.info("Generating soft labels for distillation...")
    test_subset_ds = ToxicDataset(test_subset, tokenizer, Config.max_len, is_test=True)
    test_subset_loader = DataLoader(
        test_subset_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    soft_preds = inference_fn(test_subset_loader, teacher_model, Config.device)

    # Prepare Distillation Data
    distill_df = test_subset.copy()
    distill_df[Config.target_cols] = soft_preds
    combined_df = pd.concat([train_df, distill_df], axis=0).reset_index(drop=True)

    # Cleanup Teacher
    del teacher_model, optimizer, scheduler, train_loader, test_subset_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # 5. Stage 3: Student Self-Distillation
    # ==========================================
    logger.info("\n=== Stage 3: Student Self-Distillation ===")

    student_ds = ToxicDataset(combined_df, tokenizer, Config.max_len)
    student_loader = DataLoader(
        student_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    student_model = DeepSupervisedModel(pretrained=True).to(Config.device)
    # Load DAPT weights again for Student
    student_model.model.load_state_dict(torch.load(dapt_backbone_path), strict=False)

    optimizer = AdamW(
        student_model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        total_steps=len(student_loader) * Config.epochs,
        pct_start=Config.pct_start,
    )
    awp = (
        AWP(student_model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)
        if Config.use_awp
        else None
    )

    # Train Student
    final_preds = None
    final_targets = None

    for epoch in range(Config.epochs):
        train_fn(
            0,
            student_loader,
            student_model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            Config.device,
            awp,
            logger,
        )
        val_loss, final_preds, final_targets = valid_fn(
            val_loader, student_model, criterion, Config.device, logger
        )

    # ==========================================
    # 6. Evaluation & Failure Analysis
    # ==========================================
    logger.info("\n=== Final Evaluation ===")

    final_metric = get_score(final_targets, final_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    errors = np.abs(final_targets - final_preds).mean(axis=1)

    # Calculate correlation with text length
    # Note: val_loader iterates sequentially, so order matches val_df
    val_lengths = val_df["comment_text"].str.len().values

    if len(val_lengths) == len(errors):
        correlation = np.corrcoef(errors, val_lengths)[0, 1]
        print(f"Correlation between Error and Text Length: {correlation}")
    else:
        logger.warning("Shape mismatch in failure analysis, skipping correlation.")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.9920879090652149

    if final_metric > THRESHOLD:
        logger.info(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")

        full_test_ds = ToxicDataset(test_df, tokenizer, Config.max_len, is_test=True)
        full_test_loader = DataLoader(
            full_test_ds,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds = inference_fn(full_test_loader, student_model, Config.device)

        submission = pd.read_csv(
            os.path.join(Config.input_dir, "sample_submission.csv")
        )
        # Ensure ID alignment
        submission = pd.merge(submission[["id"]], test_df[["id"]], on="id", how="left")
        # Assign predictions
        submission[Config.target_cols] = test_preds

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
