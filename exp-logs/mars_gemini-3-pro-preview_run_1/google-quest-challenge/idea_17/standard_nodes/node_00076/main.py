import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from scipy.stats import spearmanr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, AverageMeter, compute_spearman_correlation
from library.dataset import load_data, StackExchangeDataset, Collate
from library.model import DistilRoBERTaDualEncoder
from library.engine import Engine


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Ensure directories exist
    Config.create_dirs()

    # 2. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 3. Prepare Datasets and Loaders
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    collate_fn = Collate(tokenizer)

    train_dataset = StackExchangeDataset(train_df, tokenizer, is_test=False)
    val_dataset = StackExchangeDataset(val_df, tokenizer, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = DistilRoBERTaDualEncoder()
    model.to(device)

    # 5. Optimizer and Scheduler
    # We use the helper from Engine to get the parameter groups
    engine = Engine()
    optimizer_parameters = engine.get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_parameters)

    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    # 6. Training Loop
    best_score = -1.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Specific Schedule: Freeze backbone in Epoch 0, Unfreeze later
        if epoch == 0:
            print(f"\nEpoch {epoch+1}: Freezing backbone. Training Head only.")
            model.freeze_backbone()
        elif epoch == 1:
            print(f"\nEpoch {epoch+1}: Unfreezing backbone. Full Fine-tuning.")
            model.unfreeze_backbone()
        else:
            print(f"\nEpoch {epoch+1}: Full Fine-tuning.")

        # Train one epoch
        # We can reuse the engine's train_one_epoch method as it is self-contained
        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch
        )

        # Validate
        val_score = engine.validate(model, val_loader)

        # Save Best Model
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving Model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 7. Final Validation & Failure Analysis
    print("\nLoading best model for analysis...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get predictions on validation set
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            logits = model(
                q_input_ids=batch["q_input_ids"],
                q_attention_mask=batch["q_attention_mask"],
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
            )
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Compute Final Metric
    final_metric = compute_spearman_correlation(preds, targets)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample (averaged across 30 targets)
    # Shape: (N_val, )
    sample_errors = np.mean(np.abs(preds - targets), axis=1)

    # Extract features for correlation
    # We need to ensure the order matches. val_loader is shuffle=False, so index matches val_df
    val_df_analysis = val_df.copy().reset_index(drop=True)

    # Calculate text lengths
    val_df_analysis["q_title_len"] = (
        val_df_analysis["question_title"].fillna("").str.len()
    )
    val_df_analysis["q_body_len"] = (
        val_df_analysis["question_body"].fillna("").str.len()
    )
    val_df_analysis["a_len"] = val_df_analysis["answer"].fillna("").str.len()

    # Compute correlations
    features_to_check = ["q_title_len", "q_body_len", "a_len"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features_to_check:
        feat_values = val_df_analysis[feat].values
        # Ensure lengths match (just in case of drop_last issues, though val loader shouldn't drop)
        min_len = min(len(sample_errors), len(feat_values))
        corr, _ = spearmanr(sample_errors[:min_len], feat_values[:min_len])
        print(f"  Error vs {feat}: {corr:.4f}")

    # 8. Conditional Submission
    THRESHOLD = 0.40826991743732177

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        # Use Engine's predict method which handles test set loading and file saving
        engine.predict(Config.BEST_MODEL_PATH, test_df)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
