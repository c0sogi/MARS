import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, logging
from scipy.stats import spearmanr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import load_and_preprocess_data, StackExchangeDataset, CollateFn
from library.model import DualDistilRoBERTa
from library import engine


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppress transformers logging
    logging.set_verbosity_error()

    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("Loading and preprocessing data...")
    train_df = load_and_preprocess_data("train", load_cached_data=True)
    val_df = load_and_preprocess_data("val", load_cached_data=True)
    test_df = load_and_preprocess_data("test", load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Create Datasets
    train_dataset = StackExchangeDataset(train_df, tokenizer, is_test=False)
    val_dataset = StackExchangeDataset(val_df, tokenizer, is_test=False)
    test_dataset = StackExchangeDataset(test_df, tokenizer, is_test=True)

    # Create DataLoaders
    collate_fn = CollateFn(tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DualDistilRoBERTa()
    model.to(device)

    # 4. Optimizer & Scheduler Setup
    # Differential Learning Rates & Weight Decay Grouping

    # Separate parameters into backbone and head
    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.head.named_parameters()) + list(
        model.fusion_norm.named_parameters()
    )

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        # Backbone - Weight Decay
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_BACKBONE,
        },
        # Backbone - No Weight Decay
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": Config.LR_BACKBONE,
        },
        # Head - Weight Decay
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_HEAD,
        },
        # Head - No Weight Decay
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": Config.LR_HEAD,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Config.EPS)

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    print("Starting training...")
    best_score = -1.0

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        val_loss, val_score = engine.validate(model, val_loader, device)

        if val_score > best_score:
            print(
                f"Score improved from {best_score:.4f} to {val_score:.4f}. Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            print(f"Score did not improve from {best_score:.4f}.")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # Re-run validation to get predictions for analysis and final metric printing
    print("Running validation on best model...")
    # We need the raw predictions and labels, so we'll adapt the validate logic slightly here
    # or just use the engine.validate which prints the score, but we need the exact value for the check.

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    final_metric = compute_metric(all_labels, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(all_labels - all_preds), axis=1)

    # Get input lengths from dataframe
    # Note: The order in DataLoader (shuffle=False) matches the dataframe
    q_lengths = val_df["question_input"].str.len().values
    a_lengths = val_df["answer_input"].str.len().values

    # Compute correlations
    corr_q, _ = spearmanr(mae_per_sample, q_lengths)
    corr_a, _ = spearmanr(mae_per_sample, a_lengths)

    print(f"Correlation between Error Magnitude and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error Magnitude and Answer Length: {corr_a:.4f}")

    # 7. Submission
    THRESHOLD = 0.40802662717842303

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
